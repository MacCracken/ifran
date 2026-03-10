<script>
  import "../app.css";

  let { children } = $props();
</script>

<div class="app">
  <nav class="sidebar">
    <div class="logo">
      <h2>Synapse</h2>
    </div>
    <ul class="nav-links">
      <li><a href="/models">Models</a></li>
      <li><a href="/chat">Chat</a></li>
      <li><a href="/training">Training</a></li>
      <li><a href="/settings">Settings</a></li>
    </ul>
  </nav>

  <main class="content">
    {@render children()}
  </main>
</div>

<style>
  .app {
    display: flex;
    min-height: 100vh;
  }

  .sidebar {
    width: 220px;
    background: var(--bg-secondary);
    border-right: 1px solid var(--border);
    padding: 1rem;
    display: flex;
    flex-direction: column;
    gap: 2rem;
  }

  .logo h2 {
    color: var(--accent);
  }

  .nav-links {
    list-style: none;
    display: flex;
    flex-direction: column;
    gap: 0.25rem;
  }

  .nav-links a {
    display: block;
    padding: 0.5rem 0.75rem;
    color: var(--text-secondary);
    text-decoration: none;
    border-radius: var(--radius);
    font-size: 0.9rem;
    transition: all 0.15s;
  }

  .nav-links a:hover {
    background: var(--bg-tertiary);
    color: var(--text-primary);
  }

  .content {
    flex: 1;
    padding: 2rem;
    overflow-y: auto;
  }
</style>
