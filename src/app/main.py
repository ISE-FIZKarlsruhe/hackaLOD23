from fastapi import FastAPI, Request, Response, HTTPException, BackgroundTasks, Query
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from fastapi.responses import (
    HTMLResponse,
    JSONResponse,
)
from fastapi.middleware.cors import CORSMiddleware
from .config import ORIGINS, DEBUG, DATA_PATH, DB_PATH
from tree_sitter import Language, Parser
from typing import Optional
import faiss
import httpx
import pickle, os, sys, logging, json, sqlite3
from urllib.parse import parse_qs, quote
from sentence_transformers import SentenceTransformer

PREDICATE_SBERT = "<http://hackalod/fizzy/sbert>"
PREDICATE_FTS = "<http://hackalod/fizzy/fts>"

if DEBUG == "1":
    logging.basicConfig(level=logging.DEBUG)


def odata(filename):
    filepath = os.path.join(DATA_PATH, filename)
    logging.debug(f"Opening {filepath}")
    return pickle.load(open(filepath, "rb"))


# model = SentenceTransformer("jegormeister/bert-base-dutch-cased-snli")
# FIDX = odata("faissidx_dutch.pkl")
# F_LITERALS = odata("goudatm_literals.pkl")
# X_ENDPOINT = "https://www.goudatijdmachine.nl/sparql/repositories/gtm"


model = SentenceTransformer("sentence-transformers/all-mpnet-base-v2")
FIDX = odata("faissidx_rijksmuseum.pkl")
F_LITERALS = odata("df_rijksmuseum.pkl")
X_ENDPOINT = "https://api.data.netwerkdigitaalerfgoed.nl/datasets/Rijksmuseum/collection/services/collection/sparql"

if sys.platform == "darwin":
    SPARQL = Language("/usr/local/lib/sparql.dylib", "sparql")
else:
    SPARQL = Language("/usr/local/lib/sparql.so", "sparql")

PARSER = Parser()
PARSER.set_language(SPARQL)

DB_SCHEMA = """
CREATE VIRTUAL TABLE IF NOT EXISTS literal_index USING fts5(subject UNINDEXED, predicate UNINDEXED, txt);
CREATE VIRTUAL TABLE IF NOT EXISTS literal_index_vocab USING fts5vocab('literal_index', 'row');
CREATE VIRTUAL TABLE IF NOT EXISTS literal_index_spellfix USING spellfix1;
"""
DB = sqlite3.connect(DB_PATH)
DB.enable_load_extension(True)
if sys.platform == "darwin":
    DB.load_extension("/usr/local/lib/fts5stemmer.dylib")
    DB.load_extension("/usr/local/lib/spellfix.dylib")
else:
    DB.load_extension("/usr/local/lib/spellfix")
    DB.load_extension("/usr/local/lib/fts5stemmer")


DB.executescript(DB_SCHEMA)

for row in DB.execute("SELECT count(*) FROM literal_index"):
    count = row[0]
    if count < 1:
        logging.debug("FTS not available, now making one")
        buf = [
            (row["subj"], row["pred"], row["obj"])
            for index, row in F_LITERALS.iterrows()
        ]
        DB.executemany(f"INSERT INTO literal_index VALUES (?, ?, ?)", buf)
        DB.execute(
            "INSERT INTO literal_index_spellfix(word) select term from literal_index_vocab"
        )
        DB.commit()


app = FastAPI(openapi_url="/openapi")
app.add_middleware(
    CORSMiddleware,
    allow_origins=ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
templates = Jinja2Templates(directory="templates")
app.mount("/static", StaticFiles(directory="static"), name="static")


@app.get("/", response_class=HTMLResponse, include_in_schema=False)
async def homepage(request: Request):
    return templates.TemplateResponse(
        "homepage.html",
        {"request": request},
    )


@app.post("/sparql")
async def sparql_post_sparql_query(
    request: Request,
):
    content_type = request.headers.get("content-type")
    body = await request.body()
    body = body.decode("utf8")

    if content_type.startswith("application/sparql-query"):
        return await sparql_get(request, body)
    if content_type.startswith("application/x-www-form-urlencoded"):
        params = parse_qs(body)
        return await sparql_get(request, params["query"][0])

    raise HTTPException(status_code=400, detail="This request is malformed")


@app.get("/sparql")
async def sparql_get(
    request: Request,
    query: Optional[str] = Query(None),
):
    logging.debug(f"Starting /sparql query")
    accept_header = request.headers.get("accept")
    if accept_header:
        accept_headers = [ah.strip() for ah in accept_header.split(",")]
    else:
        accept_headers = []

    q = rewrite_query(query)
    logging.debug("Query has been rewritten")
    results = await external_sparql(X_ENDPOINT, q)

    if "application/sparql-results+json" in accept_headers:
        return Response(
            json.dumps(results),
            media_type="application/sparql-results+json",
            headers={"Access-Control-Allow-Origin": "*"},
        )

    return JSONResponse(results)


def search_fts(search_str: str):
    try:
        cursor = DB.execute(
            # "SELECT distinct subject FROM literal_index WHERE txt match ?",
            "SELECT distinct subject FROM literal_index WHERE txt match (SELECT word FROM literal_index_spellfix WHERE word MATCH ? LIMIT 2)",
            (search_str,),
        )
    except sqlite3.OperationalError:
        return []
    return [row[0] for row in cursor.fetchall()]


def search_sbert(search_str: str, k_nearest: int = 20):
    logging.debug(f"Searching for: [{search_str}]")
    search_emb = model.encode([search_str], convert_to_tensor=True)
    search_vectors = search_emb.cpu().detach().numpy()
    faiss.normalize_L2(search_vectors)
    _, similarities_ids = FIDX.search(search_vectors, k=k_nearest * 2)
    logging.debug("Found results")
    return set([F_LITERALS.subj.loc[si] for si in similarities_ids[0]])


def rewrite_query(query: str):
    tree = PARSER.parse(query.encode("utf8"))
    q = SPARQL.query(
        """((triples_same_subject (var) @var (property_list (property (path_element (iri_reference) @predicate) (object_list (rdf_literal) @q_object)))) @tss (".")* @tss_dot )"""
    )
    found_vars = []
    found = False
    start_byte = end_byte = 0
    var_name = q_object = None
    predicate = None
    for n, name in q.captures(tree.root_node):
        if name == "tss":
            if start_byte > 0 and end_byte > start_byte:
                if var_name is not None and q_object is not None and found:
                    found_vars.append((start_byte, end_byte, var_name, q_object))
            start_byte = n.start_byte
            end_byte = n.end_byte
            var_name = q_object = None
            found = False
        if name == "q_object":
            q_object = n.text.decode("utf8")
        if name == "predicate" and n.text == PREDICATE_SBERT.encode("utf8"):
            predicate = PREDICATE_SBERT
            found = True
        if name == "predicate" and n.text == PREDICATE_FTS.encode("utf8"):
            predicate = PREDICATE_FTS
            found = True
        if name == "var":
            var_name = n.text.decode("utf8")
        if name == "tss_dot":
            end_byte = n.end_byte

    # If there is only one,
    if start_byte > 0 and end_byte > start_byte:
        if var_name is not None and q_object is not None and found:
            found_vars.append((start_byte, end_byte, var_name, q_object, predicate))

    if len(found_vars) > 0:
        newq = []
        query_bytes = query.encode("utf8")
        i = 0
        while i < len(query_bytes):
            c = query_bytes[i]
            in_found = False
            for start_byte, end_byte, var_name, q_object, predicate in found_vars:
                if i >= start_byte and i <= end_byte:
                    in_found = True
                    if predicate == PREDICATE_SBERT:
                        fts_results = search_sbert(q_object.strip('"'))
                    if predicate == PREDICATE_FTS:
                        fts_results = search_fts(q_object.strip('"'))
                    fts_results = " ".join(
                        [
                            f"<{fts_result}>"
                            for fts_result in fts_results
                            if not fts_result.startswith("_:")
                        ]
                    )
                    if fts_results:
                        for cc in f"VALUES {var_name} {{{fts_results}}}":
                            newq.append(cc)
                    i = end_byte
            if not in_found:
                newq.append(chr(c))
            i += 1
        newq = "".join(newq)
        query = newq
    return query


async def external_sparql(endpoint: str, query: str):
    async with httpx.AsyncClient(timeout=45) as client:
        headers = {
            "Accept": "application/sparql-results+json",
            "User-Agent": "Hack-a-LOD/2023 (https://epoz.org/ ep@epoz.org)",
        }
        data = {"query": query}
        logging.debug("SPARQL query on \n%s query=%s", endpoint, quote(query))
        logging.debug(data)
        r = await client.post(
            endpoint,
            data=data,
            headers=headers,
        )
    if r.status_code == 200:
        return r.json()
    return {"exception": r.status_code, "txt": r.text}
