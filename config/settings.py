from pydantic import Field, AliasChoices
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    openai_api_key: str = Field(
        ...,
        validation_alias=AliasChoices("OPENAI_API_KEY"),
    )
    openai_model: str = Field(
        "llama-3.3-70b",
        validation_alias=AliasChoices("OPENAI_MODEL"),
    )
    openai_fast_model: str = Field(
        "llama-3.1-8b",
        validation_alias=AliasChoices("OPENAI_FAST_MODEL"),
    )
    openai_base_url: str = Field(
        "https://api.groq.com/openai/v1",
        validation_alias=AliasChoices("OPENAI_BASE_URL"),
    )

    tavily_api_key: str = Field(
        "",
        validation_alias=AliasChoices("TAVILY_API_KEY"),
    )

    semantic_scholar_api_key: str = Field(
        "",
        validation_alias=AliasChoices("SEMANTIC_SCHOLAR_API_KEY"),
    )

    langchain_tracing_v2: bool = Field(
        False,
        validation_alias=AliasChoices("LANGCHAIN_TRACING_V2"),
    )
    langchain_api_key: str = Field(
        "",
        validation_alias=AliasChoices("LANGCHAIN_API_KEY"),
    )
    langchain_project: str = Field(
        "agentic-research-assistant",
        validation_alias=AliasChoices("LANGCHAIN_PROJECT"),
    )

    max_search_results: int = Field(
        10,
        validation_alias=AliasChoices("MAX_SEARCH_RESULTS"),
    )
    similarity_threshold: float = Field(
        0.75,
        validation_alias=AliasChoices("SIMILARITY_THRESHOLD"),
    )
    top_k_results: int = Field(
        5,
        validation_alias=AliasChoices("TOP_K_RESULTS"),
    )
    deep_research_depth: int = Field(
        2,
        validation_alias=AliasChoices("DEEP_RESEARCH_DEPTH"),
    )
    deep_research_breadth: int = Field(
        3,
        validation_alias=AliasChoices("DEEP_RESEARCH_BREADTH"),
    )

    use_web_search: bool = Field(
        True,
        validation_alias=AliasChoices("USE_WEB_SEARCH"),
    )
    use_wikipedia: bool = Field(
        True,
        validation_alias=AliasChoices("USE_WIKIPEDIA"),
    )
    use_arxiv: bool = Field(
        True,
        validation_alias=AliasChoices("USE_ARXIV"),
    )
    use_mcp: bool = Field(
        False,
        validation_alias=AliasChoices("USE_MCP"),
    )

    image_generation_enabled: bool = Field(
        False,
        validation_alias=AliasChoices("IMAGE_GENERATION_ENABLED"),
    )
    google_api_key: str = Field(
        "",
        validation_alias=AliasChoices("GOOGLE_API_KEY"),
    )

    app_env: str = Field(
        "development",
        validation_alias=AliasChoices("APP_ENV"),
    )
    log_level: str = Field(
        "INFO",
        validation_alias=AliasChoices("LOG_LEVEL"),
    )
    output_dir: str = Field(
        "outputs",
        validation_alias=AliasChoices("OUTPUT_DIR"),
    )
    log_dir: str = Field(
        "logs",
        validation_alias=AliasChoices("LOG_DIR"),
    )

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )


def get_settings() -> Settings:
    return Settings()