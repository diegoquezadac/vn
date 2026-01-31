import os
import json
import asyncio
import aiofiles
from uuid import uuid4
from typing import List, Literal, Dict, Optional
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_core.documents import Document
from langchain.chat_models import init_chat_model
from langchain_core.prompts import PromptTemplate
from jinja2 import Environment, FileSystemLoader, select_autoescape
from openai import AsyncOpenAI
from src.models import Vehicle, Resolution


def get_empty_catalog() -> dict:
    """Return an empty catalog structure."""
    return {
        "brand": [],
        "model": [],
        "submodel": [],
        "trim_level": [],
    }


async def save_json_file(filename: str, data: dict):
    async with aiofiles.open(filename, "w") as f:
        await f.write(json.dumps(data, indent=2))


class Normalizer:
    """
    Vehicle representation normalizer.
    """

    def __init__(
        self,
        catalog_path: Optional[str] = None,
        persist_directory: str = "./db",
        provider: Literal["openai"] = "openai",
        model: Literal["gpt-4.1-nano", "gpt-5-nano", "gpt-5-mini"] = "gpt-4.1-nano",
        embedding_model: Literal["text-embedding-3-small", "text-embedding-3-large"] = "text-embedding-3-small",
        k: int = 3,
    ):
        # Language model setup
        if model == "gpt-4.1-nano":
            temperature = 0
        else:
            temperature = None
        llm = init_chat_model(model, model_provider=provider, temperature=temperature)

        # Jinja2 setup
        env = Environment(
            loader=FileSystemLoader("./prompts/"), autoescape=select_autoescape()
        )

        # Information extraction setup
        with open("./samples/extraction.json", "r") as file:
            extraction_samples = json.load(file)

        extraction_template = env.get_template("extraction.j2").render(
            samples=extraction_samples, x="TEMPORARY"
        )

        # Escape literal braces for PromptTemplate (double them), then add {x} variable
        extraction_template = (
            extraction_template.replace("{", "{{")
            .replace("}", "}}")
            .replace("TEMPORARY", "{x}")
            .replace("\n\n\n", "\n\n")
        )

        extraction_prompt = PromptTemplate.from_template(template=extraction_template)

        extraction_output_model = Vehicle
        extraction_chain = extraction_prompt | llm.with_structured_output(
            extraction_output_model, include_raw=True
        )

        # Information retrieval setup
        embeddings = OpenAIEmbeddings(model=embedding_model)
        vector_store = Chroma(
            collection_name="vehicles",
            embedding_function=embeddings,
            persist_directory=persist_directory,
        )

        # Data matching setup
        with open("./samples/matching.json", "r") as file:
            matching_samples = json.load(file)

        # Catalog setup - use provided path or default
        if catalog_path is None:
            catalog_path = "./data/catalog.json"

        if os.path.exists(catalog_path):
            with open(catalog_path, "r", encoding="utf-8") as file:
                catalog = json.load(file)
        else:
            # Create empty catalog if file doesn't exist
            catalog = get_empty_catalog()
            os.makedirs(os.path.dirname(catalog_path), exist_ok=True)
            with open(catalog_path, "w", encoding="utf-8") as file:
                json.dump(catalog, file, indent=2, ensure_ascii=False)

        matching_template = env.get_template("matching.j2").render(
            x="TEMPORARY_1",
            records="TEMPORARY_2",
            attribute="TEMPORARY_3",
            samples=matching_samples,
        )

        # Escape literal braces for PromptTemplate (double them), then add template variables
        matching_template = (
            matching_template.replace("{", "{{")
            .replace("}", "}}")
            .replace("TEMPORARY_1", "{x}")
            .replace("TEMPORARY_2", "{records}")
            .replace("TEMPORARY_3", "{attribute}")
            .replace("\n\n\n", "\n\n")
        )

        matching_prompt = PromptTemplate.from_template(template=matching_template)
        resolution_output_model = Resolution
        matching_chain = matching_prompt | llm.with_structured_output(
            resolution_output_model, include_raw=True
        )

        if model == "gpt-5-nano":
            self.input_cost = 0.05
            self.cached_input_cost = 0.01
            self.output_cost = 0.40
        elif model == "gpt-5-mini":
            self.input_cost = 0.25
            self.cached_input_cost = 0.03
            self.output_cost = 2.00
        elif model == "gpt-4.1-nano":
            self.input_cost = 0.100
            self.cached_input_cost = 0.025
            self.output_cost = 0.400
        else:
            raise ValueError(
                "Invalid model. Choose between 'gpt-4.1', 'gpt-4.1-mini', and 'gpt-4.1-nano"
            )

        self.llm = llm
        self.k = k
        self.catalog = catalog
        self.catalog_path = catalog_path
        self.vector_store = vector_store
        self.matching_chain = matching_chain
        self.extraction_chain = extraction_chain
        self.embedding_model = embedding_model
        self.openai_client = AsyncOpenAI()
        self.tokens = {
            "extract": {"prompt": 0, "cached": 0, "completion": 0, "count": 0},
            "match": {"prompt": 0, "cached": 0, "completion": 0, "count": 0},
        }
        self.attributes = ["brand", "model", "submodel", "trim_level"]

    def get_cost(self) -> float:
        total_prompt = self.tokens["extract"]["prompt"] + self.tokens["match"]["prompt"]
        total_cached = self.tokens["extract"]["cached"] + self.tokens["match"]["cached"]
        total_completion = self.tokens["extract"]["completion"] + self.tokens["match"]["completion"]
        return (
            total_prompt * self.input_cost
            + total_cached * self.cached_input_cost
            + total_completion * self.output_cost
        ) / 1e6

    async def extract(self, q: str) -> dict:
        """
        Extract structured information from an open-text vehicle description.
        """

        response = await self.extraction_chain.ainvoke({"x": q})
        usage = response["raw"].response_metadata["token_usage"]
        self.tokens["extract"]["prompt"] += usage["prompt_tokens"]
        self.tokens["extract"]["cached"] += usage["prompt_tokens_details"]["cached_tokens"]
        self.tokens["extract"]["completion"] += usage["completion_tokens"]
        self.tokens["extract"]["count"] += 1
        y = response["parsed"].dict()
        return y

    async def extract_with_retry(self, q: str, max_retries: int = 3, base_delay: float = 1.0) -> dict | None:
        """
        Extract with retry logic and exponential backoff.
        Returns None if all retries fail.
        """
        for attempt in range(max_retries):
            try:
                return await self.extract(q)
            except Exception:
                if attempt < max_retries - 1:
                    delay = base_delay * (2 ** attempt)  # Exponential backoff
                    await asyncio.sleep(delay)
        return None

    async def extract_batch(self, qs: List[str], max_retries: int = 3) -> List[dict]:
        """
        Extract structured information from multiple vehicle descriptions concurrently.
        Returns a list of extraction results in the same order as input.
        Failed extractions are retried up to max_retries times with exponential backoff.
        """
        tasks = [self.extract_with_retry(q, max_retries=max_retries) for q in qs]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Process results, replacing exceptions with None
        processed = []
        for result in results:
            if isinstance(result, Exception):
                processed.append(None)
            else:
                processed.append(result)

        return processed

    async def encode(self, q: str) -> List[float]:
        """
        Encode a text string into an embedding vector using OpenAI.
        """
        response = await self.openai_client.embeddings.create(
            input=q,
            model=self.embedding_model,
        )
        return response.data[0].embedding

    async def retrieve(self, q: str) -> Dict[str, List[Document]]:
        """
        Retrieve the top-k most similar values for a given attribute based on an open-text vehicle description.
        This version encodes the query internally (for backward compatibility).
        """
        embedding = await self.encode(q)
        return await self.retrieve_by_vector(embedding)

    async def retrieve_by_vector(self, embedding: List[float]) -> Dict[str, List[Document]]:
        """
        Retrieve the top-k most similar values for each attribute using a pre-computed embedding vector.
        """
        tasks = [
            self.vector_store.asimilarity_search_by_vector(
                embedding,
                k=self.k,
                filter={"attribute": attr},
            )
            for attr in self.attributes
        ]

        results = await asyncio.gather(*tasks)
        docs_by_attr = {attr: results[i] for i, attr in enumerate(self.attributes)}

        return docs_by_attr

    async def match(self, y: str, D: List[Document], attr: str) -> str:
        """
        Match an extracted value against a list of retrieved values.
        If no match is found, return an empty string.
        """

        D = [doc.page_content.removeprefix(f"{attr} ") for doc in D]
        response = await self.matching_chain.ainvoke(
            {"x": y, "records": D, "attribute": attr}
        )
        usage = response["raw"].response_metadata["token_usage"]

        self.tokens["match"]["prompt"] += usage["prompt_tokens"]
        self.tokens["match"]["cached"] += usage["prompt_tokens_details"]["cached_tokens"]
        self.tokens["match"]["completion"] += usage["completion_tokens"]
        self.tokens["match"]["count"] += 1

        return response["parsed"].result

    async def forward(self, qs: List[str]) -> List[dict]:
        """
        Normalize a list of open-text vehicle descriptions.
        For each description, the following steps are performed:
        1. Extract structured information.
        2. Retrieve the top-k most similar values for each attribute.
        3. Match the extracted values against the retrieved values.
        4. Update the catalog and vector store if necessary.
        """

        logs = []

        extraction_tasks = [self.extract(q) for q in qs]
        retrieval_tasks = [self.retrieve(q) for q in qs]

        extraction_futures = [asyncio.create_task(task) for task in extraction_tasks]
        retrieval_futures = [asyncio.create_task(task) for task in retrieval_tasks]

        for index, (extraction_future, retrieval_future) in enumerate(
            zip(extraction_futures, retrieval_futures)
        ):
            extraction_result = await extraction_future
            retrieval_result = await retrieval_future

            vehicle = extraction_result.copy()
            operations = []
            matching_results = {}

            attributes_to_match = [
                attr
                for attr in self.attributes
                if extraction_result[attr]
                and extraction_result[attr] not in self.catalog[attr]
            ]

            if attributes_to_match:
                docs_to_insert = []
                matching_tasks = [
                    self.match(extraction_result[attr], retrieval_result[attr], attr)
                    for attr in attributes_to_match
                ]
                matching_futures = [
                    asyncio.create_task(task) for task in matching_tasks
                ]

                for attr, matching_future in zip(attributes_to_match, matching_futures):
                    matching_result = await matching_future
                    matching_results[attr] = matching_result

                    if matching_result:
                        operations.append(
                            {
                                "operation": "update",
                                "attribute": attr,
                                "value": matching_result,
                            }
                        )
                        vehicle[attr] = matching_result
                    else:
                        operations.append(
                            {
                                "operation": "insert",
                                "attribute": attr,
                                "value": extraction_result[attr],
                            }
                        )
                        document = Document(
                            page_content=f"{attr} {extraction_result[attr]}",
                            metadata={"attribute": attr},
                        )
                        docs_to_insert.append(document)
                        self.catalog[attr].append(extraction_result[attr])

                if docs_to_insert:
                    uuids = [str(uuid4()) for _ in range(len(docs_to_insert))]
                    await self.vector_store.aadd_documents(
                        documents=docs_to_insert, ids=uuids
                    )

            logs.append(
                {
                    "query": qs[index],
                    "vehicle": vehicle,
                    "extraction": extraction_result,
                    "retrieval": retrieval_result,
                    "matching": matching_results,
                    "operations": operations,
                }
            )

        has_inserts = any(
            op["operation"] == "insert"
            for log in logs
            for op in log["operations"]
        )
        if has_inserts:
            with open(self.catalog_path, "w", encoding="utf-8") as file:
                json.dump(self.catalog, file, indent=2, ensure_ascii=False)

        return logs

    async def __call__(self, xs: List[str]) -> dict:
        return await self.forward(xs)
