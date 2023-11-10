# HackaLOD 2023

[HackaLOD Gouda 2023](https://netwerkdigitaalerfgoed.nl/nieuws/hackalod-2023-dit-zijn-de-toepassingen-met-erfgoeddata-voor-publiek-en-collegas/) .

Intro to [teams](https://www.instagram.com/p/CzTNa4JpvDk/)

---

To run this on your computer, first do:

`docer build -t fizzysearch .`

And then you can run it with something like this:

`docker run --rm -it -v $(pwd):/data -e X_ENDPOINT=https://someplace/sparql -e DATA_PATH=/data -p 8000:8000 fizzysearch`

At the moment you still need the data files for the semantic search in the form of pickle files to also be present in the cwd.

## TODO

- [ ] The data files are generated and cached automatically at the first run of the system.

- [ ] Add Github action to build dockerfile

- [ ] Refactor core functionality to separate Python package on PyPI
