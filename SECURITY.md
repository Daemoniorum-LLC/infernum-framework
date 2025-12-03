# Security Policy

## Supported Versions

| Version | Supported          |
| ------- | ------------------ |
| 0.1.x   | :white_check_mark: |

## Reporting a Vulnerability

If you discover a security vulnerability in Infernum, please report it responsibly:

1. **Do not** open a public GitHub issue
2. Email security concerns to: engineering@daemoniorum.com
3. Include:
   - Description of the vulnerability
   - Steps to reproduce
   - Potential impact
   - Suggested fix (if any)

## Response Timeline

- **Acknowledgment**: Within 48 hours
- **Initial Assessment**: Within 7 days
- **Resolution Timeline**: Depends on severity

## Security Considerations

Infernum runs LLMs locally. Key security aspects:

### Model Loading
- Models are loaded from HuggingFace Hub or local paths
- Verify model sources before loading untrusted models
- GGUF and SafeTensor formats are used for model weights

### API Server
- The HTTP server binds to `0.0.0.0` by default
- Consider firewall rules in production deployments
- No authentication is built-in; add a reverse proxy for production

### Local Execution
- All inference runs locally on your machine
- No data is sent to external services
- Model weights are cached in `~/.cache/huggingface/`

## Best Practices

1. Run behind a reverse proxy in production
2. Use environment variables for sensitive configuration
3. Keep dependencies updated
4. Monitor for security advisories in dependencies
