//! Test server helpers for integration testing
//!
//! Provides utilities for spinning up test servers and making requests.

use std::net::SocketAddr;
use std::time::Duration;

use axum::Router;
use tokio::net::TcpListener;

/// A test server that can be used for integration tests
pub struct TestServer {
    addr: SocketAddr,
    shutdown_tx: Option<tokio::sync::oneshot::Sender<()>>,
}

impl TestServer {
    /// Start a new test server with the given router
    pub async fn start(router: Router) -> Self {
        let listener = TcpListener::bind("127.0.0.1:0")
            .await
            .expect("Failed to bind test server");
        let addr = listener.local_addr().expect("Failed to get local address");

        let (shutdown_tx, shutdown_rx) = tokio::sync::oneshot::channel();

        tokio::spawn(async move {
            axum::serve(listener, router)
                .with_graceful_shutdown(async {
                    let _ = shutdown_rx.await;
                })
                .await
                .expect("Server error");
        });

        // Give the server a moment to start
        tokio::time::sleep(Duration::from_millis(10)).await;

        Self {
            addr,
            shutdown_tx: Some(shutdown_tx),
        }
    }

    /// Get the address the server is listening on
    pub fn addr(&self) -> SocketAddr {
        self.addr
    }

    /// Get the base URL for the server
    pub fn url(&self) -> String {
        format!("http://{}", self.addr)
    }

    /// Build a URL for a specific path
    pub fn url_for(&self, path: &str) -> String {
        let path = if path.starts_with('/') {
            path.to_string()
        } else {
            format!("/{}", path)
        };
        format!("{}{}", self.url(), path)
    }

    /// Shutdown the server
    pub fn shutdown(mut self) {
        if let Some(tx) = self.shutdown_tx.take() {
            let _ = tx.send(());
        }
    }
}

impl Drop for TestServer {
    fn drop(&mut self) {
        if let Some(tx) = self.shutdown_tx.take() {
            let _ = tx.send(());
        }
    }
}

/// Request builder for test requests
pub struct TestRequest {
    method: String,
    url: String,
    headers: Vec<(String, String)>,
    body: Option<serde_json::Value>,
}

impl TestRequest {
    /// Create a new GET request
    pub fn get(url: &str) -> Self {
        Self {
            method: "GET".to_string(),
            url: url.to_string(),
            headers: Vec::new(),
            body: None,
        }
    }

    /// Create a new POST request
    pub fn post(url: &str) -> Self {
        Self {
            method: "POST".to_string(),
            url: url.to_string(),
            headers: Vec::new(),
            body: None,
        }
    }

    /// Add a header to the request
    pub fn header(mut self, name: &str, value: &str) -> Self {
        self.headers.push((name.to_string(), value.to_string()));
        self
    }

    /// Add an Authorization header with a Bearer token
    pub fn bearer_token(self, token: &str) -> Self {
        self.header("Authorization", &format!("Bearer {}", token))
    }

    /// Set the JSON body
    pub fn json(mut self, body: serde_json::Value) -> Self {
        self.body = Some(body);
        self
    }

    /// Get the method
    pub fn method(&self) -> &str {
        &self.method
    }

    /// Get the URL
    pub fn url(&self) -> &str {
        &self.url
    }

    /// Get the headers
    pub fn headers(&self) -> &[(String, String)] {
        &self.headers
    }

    /// Get the body
    pub fn body(&self) -> Option<&serde_json::Value> {
        self.body.as_ref()
    }
}

/// Response from a test request
#[derive(Debug)]
pub struct TestResponse {
    /// HTTP status code
    pub status: u16,
    /// Response headers
    pub headers: Vec<(String, String)>,
    /// Response body as JSON (if parseable)
    pub json: Option<serde_json::Value>,
    /// Raw response body
    pub body: String,
}

impl TestResponse {
    /// Check if the response was successful (2xx)
    pub fn is_success(&self) -> bool {
        (200..300).contains(&self.status)
    }

    /// Check if a specific header is present
    pub fn has_header(&self, name: &str) -> bool {
        self.headers
            .iter()
            .any(|(k, _)| k.eq_ignore_ascii_case(name))
    }

    /// Get a header value
    pub fn header(&self, name: &str) -> Option<&str> {
        self.headers
            .iter()
            .find(|(k, _)| k.eq_ignore_ascii_case(name))
            .map(|(_, v)| v.as_str())
    }

    /// Get the JSON body, panicking if not present
    pub fn json(&self) -> &serde_json::Value {
        self.json.as_ref().expect("Response body is not valid JSON")
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use axum::{routing::get, Json};

    async fn health_handler() -> Json<serde_json::Value> {
        Json(serde_json::json!({"status": "ok"}))
    }

    #[tokio::test]
    async fn test_server_starts_and_stops() {
        let router = Router::new().route("/health", get(health_handler));
        let server = TestServer::start(router).await;

        assert!(!server.url().is_empty());
        assert_eq!(
            server.url_for("/health"),
            format!("{}/health", server.url())
        );

        server.shutdown();
    }

    #[test]
    fn test_request_builder() {
        let request = TestRequest::post("/v1/chat/completions")
            .bearer_token("sk-test")
            .json(serde_json::json!({"model": "test"}));

        assert_eq!(request.method(), "POST");
        assert!(request
            .headers()
            .iter()
            .any(|(k, v)| k == "Authorization" && v.contains("sk-test")));
        assert!(request.body().is_some());
    }
}
