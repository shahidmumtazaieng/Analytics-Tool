"""
Flexible provider configuration for LLM and embedding models.
Configured for Anthropic Claude API as primary provider for optimal MCP integration.
"""

import os
from typing import Optional, Union
from pydantic_ai.providers.anthropic import AnthropicProvider
from pydantic_ai.models.anthropic import AnthropicModel
from pydantic_ai.providers.openai import OpenAIProvider
from pydantic_ai.models.openai import OpenAIModel
import openai
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


def get_llm_model(model_choice: Optional[str] = None) -> Union[AnthropicModel, OpenAIModel]:
    """
    Get LLM model configuration based on environment variables.
    Configured for Anthropic Claude API as primary provider for optimal MCP integration.

    Args:
        model_choice: Optional override for model choice

    Returns:
        Configured Anthropic or fallback model
    """
    # Determine provider preference
    llm_provider = os.getenv('LLM_PROVIDER', 'anthropic')

    if llm_provider == 'anthropic':
        return get_anthropic_model(model_choice)
    elif llm_provider == 'gemini':
        return get_gemini_model(model_choice)
    elif llm_provider == 'openai':
        return get_openai_model(model_choice)
    else:
        # Default to Anthropic for best MCP support
        return get_anthropic_model(model_choice)


def get_anthropic_model(model_choice: Optional[str] = None) -> AnthropicModel:
    """
    Get Anthropic Claude model configuration.
    Optimal for MCP (Model Context Protocol) integration.

    Args:
        model_choice: Optional override for model choice

    Returns:
        Configured Anthropic Claude model
    """
    # Use Claude 3.5 Sonnet as default for best MCP performance
    llm_choice = model_choice or os.getenv('ANTHROPIC_MODEL', 'claude-3-5-sonnet-20241022')
    api_key = os.getenv('ANTHROPIC_API_KEY', '')

    if not api_key:
        raise ValueError("ANTHROPIC_API_KEY environment variable is required for Anthropic provider")

    provider = AnthropicProvider(api_key=api_key)
    return AnthropicModel(llm_choice, provider=provider)


def get_gemini_model(model_choice: Optional[str] = None) -> OpenAIModel:
    """
    Get Gemini model configuration (fallback option).

    Args:
        model_choice: Optional override for model choice

    Returns:
        Configured Gemini-compatible model
    """
    llm_choice = model_choice or os.getenv('GEMINI_MODEL', 'gemini-1.5-flash')
    base_url = os.getenv('GEMINI_BASE_URL', 'https://generativelanguage.googleapis.com/v1beta')
    api_key = os.getenv('GEMINI_API_KEY', '')

    if not api_key:
        raise ValueError("GEMINI_API_KEY environment variable is required for Gemini provider")

    provider = OpenAIProvider(base_url=base_url, api_key=api_key)
    return OpenAIModel(llm_choice, provider=provider)


def get_openai_model(model_choice: Optional[str] = None) -> OpenAIModel:
    """
    Get OpenAI model configuration (fallback option).

    Args:
        model_choice: Optional override for model choice

    Returns:
        Configured OpenAI model
    """
    llm_choice = model_choice or os.getenv('OPENAI_MODEL', 'gpt-4-turbo-preview')
    api_key = os.getenv('OPENAI_API_KEY', '')

    if not api_key:
        raise ValueError("OPENAI_API_KEY environment variable is required for OpenAI provider")

    provider = OpenAIProvider(api_key=api_key)
    return OpenAIModel(llm_choice, provider=provider)


def get_embedding_client() -> openai.AsyncOpenAI:
    """
    Get embedding client configuration based on environment variables.
    Supports Anthropic, Gemini, and OpenAI embeddings.

    Returns:
        Configured client for embeddings
    """
    embedding_provider = os.getenv('EMBEDDING_PROVIDER', 'openai')

    if embedding_provider == 'anthropic':
        # Anthropic doesn't have embeddings API, fallback to OpenAI
        base_url = os.getenv('OPENAI_BASE_URL', 'https://api.openai.com/v1')
        api_key = os.getenv('OPENAI_API_KEY', '')
    elif embedding_provider == 'gemini':
        base_url = os.getenv('GEMINI_BASE_URL', 'https://generativelanguage.googleapis.com/v1beta')
        api_key = os.getenv('GEMINI_API_KEY', '')
    else:  # openai
        base_url = os.getenv('OPENAI_BASE_URL', 'https://api.openai.com/v1')
        api_key = os.getenv('OPENAI_API_KEY', '')

    if not api_key:
        raise ValueError(f"API key required for {embedding_provider} embeddings")

    return openai.AsyncOpenAI(
        base_url=base_url,
        api_key=api_key
    )


def get_embedding_model() -> str:
    """
    Get embedding model name from environment.

    Returns:
        Embedding model name
    """
    embedding_provider = os.getenv('EMBEDDING_PROVIDER', 'openai')

    if embedding_provider == 'anthropic':
        # Anthropic doesn't have embeddings, use OpenAI
        return os.getenv('EMBEDDING_MODEL', 'text-embedding-3-small')
    elif embedding_provider == 'gemini':
        return os.getenv('EMBEDDING_MODEL', 'text-embedding-004')
    else:  # openai
        return os.getenv('EMBEDDING_MODEL', 'text-embedding-3-small')


def get_ingestion_model() -> Union[AnthropicModel, OpenAIModel]:
    """
    Get ingestion-specific LLM model (can be faster/cheaper than main model).

    Returns:
        Configured model for ingestion tasks
    """
    ingestion_choice = os.getenv('INGESTION_LLM_CHOICE')

    # If no specific ingestion model, use the main model
    if not ingestion_choice:
        return get_llm_model()

    return get_llm_model(model_choice=ingestion_choice)


def get_mcp_model() -> Union[AnthropicModel, OpenAIModel]:
    """
    Get MCP-optimized model configuration.
    Anthropic Claude is preferred for MCP integration.

    Returns:
        Configured model optimized for MCP
    """
    mcp_provider = os.getenv('MCP_PROVIDER', 'anthropic')
    mcp_model = os.getenv('MCP_MODEL')

    if mcp_provider == 'anthropic':
        return get_anthropic_model(mcp_model or 'claude-3-5-sonnet-20241022')
    elif mcp_provider == 'openai':
        return get_openai_model(mcp_model or 'gpt-4-turbo-preview')
    else:
        # Default to Anthropic for best MCP support
        return get_anthropic_model(mcp_model or 'claude-3-5-sonnet-20241022')


# Provider information functions
def get_llm_provider() -> str:
    """Get the LLM provider name."""
    return os.getenv('LLM_PROVIDER', 'anthropic')


def get_embedding_provider() -> str:
    """Get the embedding provider name."""
    return os.getenv('EMBEDDING_PROVIDER', 'openai')


def get_mcp_provider() -> str:
    """Get the MCP provider name."""
    return os.getenv('MCP_PROVIDER', 'anthropic')


def validate_configuration() -> bool:
    """
    Validate that required environment variables are set.

    Returns:
        True if configuration is valid
    """
    required_vars = []

    # Check LLM configuration
    llm_provider = get_llm_provider()
    if llm_provider == 'anthropic':
        required_vars.extend(['ANTHROPIC_API_KEY'])
    elif llm_provider == 'gemini':
        required_vars.extend(['GEMINI_API_KEY'])
    elif llm_provider == 'openai':
        required_vars.extend(['OPENAI_API_KEY'])

    # Check embedding configuration
    embedding_provider = get_embedding_provider()
    if embedding_provider == 'openai':
        if 'OPENAI_API_KEY' not in required_vars:
            required_vars.append('OPENAI_API_KEY')
    elif embedding_provider == 'gemini':
        if 'GEMINI_API_KEY' not in required_vars:
            required_vars.append('GEMINI_API_KEY')

    # Check MCP configuration
    mcp_provider = get_mcp_provider()
    if mcp_provider == 'anthropic' and 'ANTHROPIC_API_KEY' not in required_vars:
        required_vars.append('ANTHROPIC_API_KEY')

    missing_vars = [var for var in required_vars if not os.getenv(var)]

    if missing_vars:
        raise ValueError(f"Missing required environment variables: {missing_vars}")

    return True


def get_model_info() -> dict:
    """
    Get information about configured models.

    Returns:
        Dictionary with model configuration info
    """
    llm_provider = get_llm_provider()

    return {
        "llm_provider": llm_provider,
        "llm_model": os.getenv(f'{llm_provider.upper()}_MODEL', 'claude-3-5-sonnet-20241022'),
        "embedding_provider": get_embedding_provider(),
        "embedding_model": get_embedding_model(),
        "mcp_provider": get_mcp_provider(),
        "mcp_optimized": llm_provider == 'anthropic'
    }


def get_provider_capabilities() -> dict:
    """
    Get capabilities of current provider configuration.

    Returns:
        Dictionary with provider capabilities
    """
    llm_provider = get_llm_provider()

    return {
        "mcp_support": llm_provider == 'anthropic',
        "function_calling": True,
        "streaming": True,
        "vision": llm_provider in ['anthropic', 'openai'],
        "embeddings": get_embedding_provider() in ['openai', 'gemini'],
        "cost_tier": {
            'anthropic': 'premium',
            'openai': 'premium',
            'gemini': 'budget'
        }.get(llm_provider, 'unknown')
    }
