<script lang="ts">
  import { onMount } from "svelte";
  import { models, loading, error, refreshModels, removeModel } from "$lib/stores/models";

  onMount(refreshModels);

  function formatBytes(bytes: number): string {
    if (bytes >= 1e9) return (bytes / 1e9).toFixed(1) + ' GB';
    if (bytes >= 1e6) return (bytes / 1e6).toFixed(1) + ' MB';
    return bytes + ' B';
  }

  async function handleDelete(id: string, name: string) {
    if (confirm(`Delete model "${name}"?`)) {
      await removeModel(id);
    }
  }
</script>

<div class="models-page">
  <div class="header">
    <h1>Models</h1>
    <button class="btn btn-primary" onclick={refreshModels}>Refresh</button>
  </div>

  {#if $error}
    <div class="card" style="border-color: var(--error)">
      <p style="color: var(--error)">{$error}</p>
    </div>
  {/if}

  {#if $loading}
    <p style="color: var(--text-secondary)">Loading...</p>
  {:else if $models.length === 0}
    <div class="card">
      <p>No models found. Pull a model with <code>synapse pull</code>.</p>
    </div>
  {:else}
    <div class="model-list">
      {#each $models as model}
        <div class="card model-card">
          <div class="model-info">
            <h3>{model.name}</h3>
            <div class="meta">
              <span class="badge badge-info">{model.format}</span>
              <span class="badge">{model.quant}</span>
              <span>{formatBytes(model.size_bytes)}</span>
              {#if model.architecture}
                <span>{model.architecture}</span>
              {/if}
            </div>
            {#if model.repo_id}
              <p class="repo">{model.repo_id}</p>
            {/if}
          </div>
          <div class="model-actions">
            <button class="btn btn-danger" onclick={() => handleDelete(model.id, model.name)}>
              Delete
            </button>
          </div>
        </div>
      {/each}
    </div>
  {/if}
</div>

<style>
  .models-page {
    display: flex;
    flex-direction: column;
    gap: 1rem;
  }

  .header {
    display: flex;
    justify-content: space-between;
    align-items: center;
  }

  .model-list {
    display: flex;
    flex-direction: column;
    gap: 0.75rem;
  }

  .model-card {
    display: flex;
    justify-content: space-between;
    align-items: center;
  }

  .model-info {
    display: flex;
    flex-direction: column;
    gap: 0.375rem;
  }

  .meta {
    display: flex;
    gap: 0.5rem;
    align-items: center;
    color: var(--text-secondary);
    font-size: 0.85rem;
  }

  .badge {
    background: var(--bg-tertiary);
    border: 1px solid var(--border);
  }

  .repo {
    color: var(--text-secondary);
    font-size: 0.8rem;
    font-family: var(--mono);
  }
</style>
