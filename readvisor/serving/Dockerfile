# Serving image for the ReAdvisor mBART response generator.
# Base pinned to match the 2021 stack (torch 1.6 / spaCy 2.3 / transformers fork).
#
# Build from the REPO ROOT so pyproject.toml + the readvisor/ package are in the
# build context, e.g.:
#     docker build -m 14g -f readvisor/serving/Dockerfile -t respogen .
FROM python:3.8.5

WORKDIR /app

# Copy the package metadata + source. Serving runtime deps are declared in
# pyproject.toml [project].dependencies and installed via `pip install .`.
COPY pyproject.toml README.md ./
COPY readvisor ./readvisor

RUN pip install --upgrade pip && pip install .

# NOTE: the spaCy NER models and the mBART checkpoint are NOT baked into the
# image. They are provided at runtime via a mounted volume / model dir and are
# referenced by the relative paths in readvisor/serving/config.json (as before).
CMD ["python", "-m", "readvisor.serving.main", "readvisor/serving/config.json"]
