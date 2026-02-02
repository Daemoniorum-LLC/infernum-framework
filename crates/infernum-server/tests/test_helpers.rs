//! Test utilities for infernum-server integration tests.

use std::net::SocketAddr;
use std::time::Duration;

use axum::Router;
use tokio::net::TcpListener;
use tokio::sync::oneshot;

/// Test server for integration testing.
pub struct TestServer {
    pub addr: SocketAddr,
    client: reqwest::Client,
    shutdown_tx: Option<oneshot::Sender<()>>,
}

impl TestServer {
    /// Starts a test server with the given router.
    pub async fn start(router: Router) -> Self {
        let listener = TcpListener::bind("127.0.0.1:0")
            .await
            .expect("Failed to bind test server");
        let addr = listener.local_addr().expect("Failed to get local addr");

        let (shutdown_tx, shutdown_rx) = oneshot::channel();

        tokio::spawn(async move {
            axum::serve(listener, router)
                .with_graceful_shutdown(async {
                    let _ = shutdown_rx.await;
                })
                .await
                .expect("Server error");
        });

        tokio::time::sleep(Duration::from_millis(50)).await;

        let client = reqwest::Client::builder()
            .timeout(Duration::from_secs(30))
            .build()
            .expect("Failed to build client");

        Self {
            addr,
            client,
            shutdown_tx: Some(shutdown_tx),
        }
    }

    pub fn url(&self, path: &str) -> String {
        format!("http://{}{}", self.addr, path)
    }

    pub async fn get(&self, path: &str) -> reqwest::Response {
        self.client
            .get(&self.url(path))
            .send()
            .await
            .expect("GET request failed")
    }

    pub async fn post_json<T: serde::Serialize>(&self, path: &str, body: &T) -> reqwest::Response {
        self.client
            .post(&self.url(path))
            .json(body)
            .header("Content-Type", "application/json")
            .send()
            .await
            .expect("POST request failed")
    }
}

impl Drop for TestServer {
    fn drop(&mut self) {
        if let Some(tx) = self.shutdown_tx.take() {
            let _ = tx.send(());
        }
    }
}

/// Helper to parse JSON response.
pub async fn json_body(response: reqwest::Response) -> serde_json::Value {
    response.json().await.expect("Failed to parse JSON")
}
