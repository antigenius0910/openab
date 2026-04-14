use crate::acp::connection::AcpConnection;
use crate::config::AgentConfig;
use anyhow::{anyhow, Result};
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::{Mutex, RwLock};
use tokio::time::Instant;
use tracing::{info, warn};

pub struct SessionPool {
    // Arc<Mutex<AcpConnection>> allows the map lock to be released before streaming begins.
    // Previously RwLock<HashMap<String, AcpConnection>> held the map write lock for the
    // entire prompt duration, serialising all sessions across all platforms.
    connections: RwLock<HashMap<String, Arc<Mutex<AcpConnection>>>>,
    config: AgentConfig,
    max_sessions: usize,
}

impl SessionPool {
    pub fn new(config: AgentConfig, max_sessions: usize) -> Self {
        Self {
            connections: RwLock::new(HashMap::new()),
            config,
            max_sessions,
        }
    }

    pub async fn get_or_create(&self, thread_id: &str) -> Result<()> {
        // Fast path: session already alive — read lock only.
        {
            let conns = self.connections.read().await;
            if let Some(entry) = conns.get(thread_id) {
                if entry.lock().await.alive() {
                    return Ok(());
                }
            }
        }

        // Slow path: need to create or rebuild — write lock briefly.
        let mut conns = self.connections.write().await;

        // Double-check after acquiring write lock.
        if let Some(entry) = conns.get(thread_id) {
            if entry.lock().await.alive() {
                return Ok(());
            }
            warn!(thread_id, "stale connection, rebuilding");
            conns.remove(thread_id);
        }

        if conns.len() >= self.max_sessions {
            return Err(anyhow!("pool exhausted ({} sessions)", self.max_sessions));
        }

        let mut conn = AcpConnection::spawn(
            &self.config.command,
            &self.config.args,
            &self.config.working_dir,
            &self.config.env,
        )
        .await?;

        conn.initialize().await?;
        conn.session_new(&self.config.working_dir).await?;

        // is_rebuild is always false here: we removed the stale entry above,
        // so contains_key cannot be true at this point.
        conns.insert(thread_id.to_string(), Arc::new(Mutex::new(conn)));
        Ok(())
    }

    /// Get exclusive access to a connection for the duration of f().
    /// The map read lock is held only for the Arc clone; streaming holds only
    /// the per-session Mutex, allowing other sessions to run concurrently.
    pub async fn with_connection<F, R>(&self, thread_id: &str, f: F) -> Result<R>
    where
        F: FnOnce(&mut AcpConnection) -> std::pin::Pin<Box<dyn std::future::Future<Output = Result<R>> + Send + '_>>,
    {
        // Read lock released as soon as we have the Arc.
        let entry = {
            let conns = self.connections.read().await;
            conns
                .get(thread_id)
                .cloned()
                .ok_or_else(|| anyhow!("no connection for thread {thread_id}"))?
        };

        // Only this session's Mutex is held during streaming.
        let mut conn = entry.lock().await;
        f(&mut conn).await
    }

    pub async fn cleanup_idle(&self, ttl_secs: u64) {
        let cutoff = Instant::now() - std::time::Duration::from_secs(ttl_secs);
        let mut conns = self.connections.write().await;
        let mut stale = Vec::new();
        for (k, entry) in conns.iter() {
            if let Ok(c) = entry.try_lock() {
                if c.last_active < cutoff || !c.alive() {
                    stale.push(k.clone());
                }
            }
        }
        for key in stale {
            info!(thread_id = %key, "cleaning up idle session");
            conns.remove(&key);
        }
    }

    pub async fn shutdown(&self) {
        let mut conns = self.connections.write().await;
        let count = conns.len();
        conns.clear(); // kill_on_drop handles process cleanup
        info!(count, "pool shutdown complete");
    }
}

#[cfg(test)]
mod lock_contention_tests {
    use std::collections::HashMap;
    use std::sync::Arc;
    use std::time::{Duration, Instant};
    use tokio::sync::RwLock;

    /// Reproduces the with_connection locking pattern from SessionPool.
    /// The write lock is acquired and held for the entire duration of f().
    async fn with_lock<F, R>(map: &RwLock<HashMap<String, u32>>, key: &str, f: F) -> R
    where
        F: FnOnce(&mut u32) -> std::pin::Pin<Box<dyn std::future::Future<Output = R> + Send + '_>>,
    {
        let mut guard = map.write().await; // write lock acquired here
        let val = guard.get_mut(key).unwrap();
        f(val).await
        // guard (and write lock) dropped here — AFTER f() completes
    }

    /// BEFORE fix: write lock held during f() → tasks run serially (~2s).
    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn test_write_lock_held_during_streaming_serial() {
        let map = Arc::new(RwLock::new(HashMap::from([
            ("discord:thread-1".to_string(), 0u32),
            ("slack:thread-2".to_string(), 0u32),
        ])));

        let start = Instant::now();

        let map1 = map.clone();
        let task1 = tokio::spawn(async move {
            with_lock(&map1, "discord:thread-1", |v| {
                Box::pin(async move {
                    tokio::time::sleep(Duration::from_secs(1)).await;
                    *v += 1;
                })
            })
            .await;
        });

        let map2 = map.clone();
        let task2 = tokio::spawn(async move {
            with_lock(&map2, "slack:thread-2", |v| {
                Box::pin(async move {
                    tokio::time::sleep(Duration::from_secs(1)).await;
                    *v += 1;
                })
            })
            .await;
        });

        let _ = tokio::join!(task1, task2);
        let elapsed = start.elapsed();
        println!("[BEFORE fix] Elapsed: {:.2}s (expected ~2s, serial)", elapsed.as_secs_f64());
        assert!(elapsed >= Duration::from_millis(1900),
            "Expected serial ~2s, got {:.2}s", elapsed.as_secs_f64());
    }

    /// AFTER fix: map lock released before f(); per-entry Mutex allows concurrent streaming (~1s).
    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn test_per_entry_mutex_allows_concurrent_streaming() {
        use tokio::sync::Mutex;

        // Simulate the fixed pool: RwLock<HashMap<String, Arc<Mutex<u32>>>>
        let map: Arc<RwLock<HashMap<String, Arc<Mutex<u32>>>>> = Arc::new(RwLock::new(
            HashMap::from([
                ("discord:thread-1".to_string(), Arc::new(Mutex::new(0u32))),
                ("slack:thread-2".to_string(), Arc::new(Mutex::new(0u32))),
            ]),
        ));

        async fn with_lock_fixed(
            map: &RwLock<HashMap<String, Arc<Mutex<u32>>>>,
            key: &str,
            f: impl FnOnce(&mut u32) -> std::pin::Pin<Box<dyn std::future::Future<Output = ()> + Send + '_>>,
        ) {
            // Map read lock released immediately after Arc clone.
            let entry = map.read().await.get(key).cloned().unwrap();
            let mut val = entry.lock().await;
            f(&mut val).await;
        }

        let start = Instant::now();

        let map1 = map.clone();
        let task1 = tokio::spawn(async move {
            with_lock_fixed(&map1, "discord:thread-1", |v| {
                Box::pin(async move {
                    tokio::time::sleep(Duration::from_secs(1)).await;
                    *v += 1;
                })
            })
            .await;
        });

        let map2 = map.clone();
        let task2 = tokio::spawn(async move {
            with_lock_fixed(&map2, "slack:thread-2", |v| {
                Box::pin(async move {
                    tokio::time::sleep(Duration::from_secs(1)).await;
                    *v += 1;
                })
            })
            .await;
        });

        let _ = tokio::join!(task1, task2);
        let elapsed = start.elapsed();
        println!("[AFTER fix]  Elapsed: {:.2}s (expected ~1s, concurrent)", elapsed.as_secs_f64());
        assert!(elapsed < Duration::from_millis(1500),
            "Expected concurrent ~1s, got {:.2}s", elapsed.as_secs_f64());
    }
}
