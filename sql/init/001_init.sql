
-- Enable vector support.
CREATE EXTENSION IF NOT EXISTS vector;

-- Create FAQ document storage.
CREATE TABLE IF NOT EXISTS documents (
    id BIGSERIAL PRIMARY KEY,
    document_key TEXT NOT NULL,
    course TEXT NOT NULL,
    section TEXT NOT NULL,
    question TEXT NOT NULL,
    answer TEXT NOT NULL,
    embedding VECTOR(384) NOT NULL,
    embedding_model TEXT NOT NULL
);

-- Prevent duplicate documents.
CREATE UNIQUE INDEX IF NOT EXISTS uq_documents_document_key
ON documents (document_key);

-- Optimize metadata filtering.
CREATE INDEX IF NOT EXISTS ix_documents_course_model
ON documents (course, embedding_model);
