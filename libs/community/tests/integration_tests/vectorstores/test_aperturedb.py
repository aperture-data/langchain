"""Test ApertureDB functionality."""

import uuid
from typing import List, Optional

import pytest
from langchain_core.documents import Document
from langchain_standard_tests.integration_tests.vectorstores import (
    AsyncReadWriteTestSuite,
    ReadWriteTestSuite,
)

from langchain_community.vectorstores import ApertureDB
from tests.integration_tests.vectorstores.fake_embeddings import (
    FakeEmbeddings,
    fake_texts,
)


class TestApertureDBReadWriteTestSuite(ReadWriteTestSuite):
    @pytest.fixture
    def vectorstore(self) -> ApertureDB:
        descriptor_set = uuid.uuid4().hex  # Fresh descriptor set for each test
        return ApertureDB(embeddings=self.get_embeddings(),
            descriptor_set=descriptor_set)


class TestAsyncApertureDBReadWriteTestSuite(AsyncReadWriteTestSuite):
    @pytest.fixture
    async def vectorstore(self) -> ApertureDB:
        descriptor_set = uuid.uuid4().hex  # Fresh descriptor set for each test
        return ApertureDB(embeddings=self.get_embeddings(),
            descriptor_set=descriptor_set)
            