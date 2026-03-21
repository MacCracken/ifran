<script>
  import { onMount } from "svelte";
  import {
    listRlhfSessions,
    createRlhfSession,
    getNextPair,
    submitAnnotation,
    getSessionStats,
    exportRlhfSession,
  } from "$lib/api/client";

  let sessions = $state([]);
  let selectedSession = $state(null);
  let currentPair = $state(null);
  let stats = $state(null);
  let newSessionName = $state("");
  let newModelName = $state("");
  let loading = $state(false);

  onMount(async () => {
    await loadSessions();
  });

  async function loadSessions() {
    try {
      const result = await listRlhfSessions();
      sessions = result.data || [];
    } catch {
      sessions = [];
    }
  }

  async function handleCreateSession() {
    if (!newSessionName || !newModelName) return;
    loading = true;
    try {
      await createRlhfSession(newSessionName, newModelName);
      newSessionName = "";
      newModelName = "";
      await loadSessions();
    } finally {
      loading = false;
    }
  }

  async function selectSession(session) {
    selectedSession = session;
    await loadPairAndStats();
  }

  async function loadPairAndStats() {
    if (!selectedSession) return;
    const [pair, s] = await Promise.all([
      getNextPair(selectedSession.id),
      getSessionStats(selectedSession.id),
    ]);
    currentPair = pair.done ? null : pair;
    stats = s;
  }

  async function handleAnnotate(preference) {
    if (!currentPair) return;
    loading = true;
    try {
      await submitAnnotation(currentPair.id, preference);
      await loadPairAndStats();
    } finally {
      loading = false;
    }
  }

  async function handleExport() {
    if (!selectedSession) return;
    const result = await exportRlhfSession(selectedSession.id);
    const blob = new Blob([JSON.stringify(result, null, 2)], { type: "application/json" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = `rlhf-export-${selectedSession.id}.json`;
    a.click();
    URL.revokeObjectURL(url);
  }
</script>

<div class="rlhf-page">
  <h1>RLHF Annotation</h1>

  <section class="sessions-panel">
    <h2>Sessions</h2>
    <div class="create-form">
      <input bind:value={newSessionName} placeholder="Session name" />
      <input bind:value={newModelName} placeholder="Model name" />
      <button onclick={handleCreateSession} disabled={loading}>Create</button>
    </div>
    <ul class="session-list">
      {#each sessions as session}
        <li>
          <button
            class="session-item"
            class:active={selectedSession?.id === session.id}
            onclick={() => selectSession(session)}
          >
            <span class="session-name">{session.name}</span>
            <span class="session-model">{session.model_name}</span>
          </button>
        </li>
      {/each}
    </ul>
  </section>

  {#if selectedSession}
    <section class="annotation-panel">
      {#if stats}
        <div class="progress-section">
          <div class="progress-bar">
            <div
              class="progress-fill"
              style="width: {stats.total_pairs > 0 ? (stats.annotated_count / stats.total_pairs) * 100 : 0}%"
            ></div>
          </div>
          <span class="progress-text">{stats.annotated_count} / {stats.total_pairs} annotated</span>
          <button class="export-btn" onclick={handleExport}>Export</button>
        </div>
      {/if}

      {#if currentPair}
        <div class="prompt-section">
          <h3>Prompt</h3>
          <p class="prompt-text">{currentPair.prompt}</p>
        </div>

        <div class="responses">
          <div class="response-card">
            <h4>Response A</h4>
            <p>{currentPair.response_a}</p>
          </div>
          <div class="response-card">
            <h4>Response B</h4>
            <p>{currentPair.response_b}</p>
          </div>
        </div>

        <div class="preference-buttons">
          <button class="pref-btn pref-a" onclick={() => handleAnnotate("response_a")} disabled={loading}>A is Better</button>
          <button class="pref-btn pref-b" onclick={() => handleAnnotate("response_b")} disabled={loading}>B is Better</button>
          <button class="pref-btn pref-tie" onclick={() => handleAnnotate("tie")} disabled={loading}>Tie</button>
          <button class="pref-btn pref-bad" onclick={() => handleAnnotate("both_bad")} disabled={loading}>Both Bad</button>
        </div>
      {:else}
        <div class="done-message">
          <p>All pairs annotated for this session.</p>
        </div>
      {/if}
    </section>
  {/if}
</div>

<style>
  .rlhf-page {
    max-width: 1200px;
  }

  h1 {
    margin-bottom: 1.5rem;
  }

  .sessions-panel {
    margin-bottom: 2rem;
  }

  .create-form {
    display: flex;
    gap: 0.5rem;
    margin-bottom: 1rem;
  }

  .create-form input {
    padding: 0.5rem;
    border: 1px solid var(--border);
    border-radius: var(--radius);
    background: var(--bg-secondary);
    color: var(--text-primary);
    flex: 1;
  }

  .create-form button {
    padding: 0.5rem 1rem;
    background: var(--accent);
    color: white;
    border: none;
    border-radius: var(--radius);
    cursor: pointer;
  }

  .session-list {
    list-style: none;
    display: flex;
    gap: 0.5rem;
    flex-wrap: wrap;
  }

  .session-item {
    padding: 0.5rem 1rem;
    border: 1px solid var(--border);
    border-radius: var(--radius);
    background: var(--bg-secondary);
    color: var(--text-primary);
    cursor: pointer;
    display: flex;
    flex-direction: column;
    gap: 0.25rem;
  }

  .session-item.active {
    border-color: var(--accent);
    background: var(--bg-tertiary);
  }

  .session-model {
    font-size: 0.8rem;
    color: var(--text-secondary);
  }

  .annotation-panel {
    border: 1px solid var(--border);
    border-radius: var(--radius);
    padding: 1.5rem;
    background: var(--bg-secondary);
  }

  .progress-section {
    display: flex;
    align-items: center;
    gap: 1rem;
    margin-bottom: 1.5rem;
  }

  .progress-bar {
    flex: 1;
    height: 8px;
    background: var(--bg-tertiary);
    border-radius: 4px;
    overflow: hidden;
  }

  .progress-fill {
    height: 100%;
    background: var(--accent);
    transition: width 0.3s;
  }

  .progress-text {
    font-size: 0.85rem;
    color: var(--text-secondary);
    white-space: nowrap;
  }

  .export-btn {
    padding: 0.4rem 0.8rem;
    background: var(--bg-tertiary);
    color: var(--text-primary);
    border: 1px solid var(--border);
    border-radius: var(--radius);
    cursor: pointer;
    font-size: 0.85rem;
  }

  .prompt-section {
    margin-bottom: 1.5rem;
  }

  .prompt-text {
    padding: 1rem;
    background: var(--bg-tertiary);
    border-radius: var(--radius);
    white-space: pre-wrap;
  }

  .responses {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 1rem;
    margin-bottom: 1.5rem;
  }

  .response-card {
    padding: 1rem;
    border: 1px solid var(--border);
    border-radius: var(--radius);
    background: var(--bg-tertiary);
  }

  .response-card h4 {
    margin-bottom: 0.5rem;
    color: var(--text-secondary);
  }

  .response-card p {
    white-space: pre-wrap;
  }

  .preference-buttons {
    display: flex;
    gap: 0.75rem;
    justify-content: center;
  }

  .pref-btn {
    padding: 0.75rem 1.5rem;
    border: 1px solid var(--border);
    border-radius: var(--radius);
    cursor: pointer;
    font-weight: 500;
    color: var(--text-primary);
    background: var(--bg-tertiary);
    transition: all 0.15s;
  }

  .pref-btn:hover:not(:disabled) {
    border-color: var(--accent);
  }

  .pref-btn:disabled {
    opacity: 0.5;
    cursor: not-allowed;
  }

  .done-message {
    text-align: center;
    padding: 2rem;
    color: var(--text-secondary);
  }
</style>
