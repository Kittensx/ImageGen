# Vendored Prompt Parser 21

`prompt_parser_fixed_v21.py` was supplied for IMAGE_GEN integration as Prompt Parser 21.

Credit:

- Contributor: GitHub user **Konpr**
- Repository: `https://github.com/Konpr/whats-new`

The vendored implementation is isolated behind:

- `modules.prompt_parsers.adapters.parser21.Parser21PromptParserAdapter`
- `modules.prompt_parsers.registry.PromptParserRegistry`

It is not imported by the legacy prompt path and is marked experimental. IMAGE_GEN does not silently substitute Prompt Parser 21 for the legacy parser or vice versa.
