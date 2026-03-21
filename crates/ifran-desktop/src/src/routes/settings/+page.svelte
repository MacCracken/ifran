<script lang="ts">
  import { onMount } from "svelte";
  import { refreshHardware, hardware, refreshStatus, status } from "$lib/stores/system";

  onMount(() => {
    refreshStatus();
    refreshHardware();
  });
</script>

<div class="settings-page">
  <h1>Settings</h1>

  <section>
    <h2>System Information</h2>

    <div class="card">
      <h3>Server</h3>
      {#if $status}
        <dl>
          <dt>Version</dt><dd>{$status.version}</dd>
          <dt>Loaded Models</dt><dd>{$status.loaded_models}</dd>
          <dt>Backends</dt><dd>{$status.registered_backends.join(", ")}</dd>
        </dl>
      {:else}
        <p style="color: var(--text-secondary)">Not connected. Make sure <code>ifran serve</code> is running.</p>
      {/if}
    </div>

    <div class="card">
      <h3>Hardware</h3>
      {#if $hardware}
        <dl>
          <dt>CPU</dt><dd>{$hardware.cpu.model} ({$hardware.cpu.cores}c/{$hardware.cpu.threads}t)</dd>
          <dt>RAM</dt><dd>{Math.round($hardware.cpu.memory_total_mb / 1024)} GB total, {Math.round($hardware.cpu.memory_available_mb / 1024)} GB free</dd>
        </dl>

        {#if $hardware.gpus.length > 0}
          <h3 style="margin-top: 1rem">GPUs</h3>
          {#each $hardware.gpus as gpu}
            <dl>
              <dt>GPU {gpu.index}</dt><dd>{gpu.name}</dd>
              <dt>VRAM</dt><dd>{gpu.memory_total_mb} MB total, {gpu.memory_free_mb} MB free</dd>
            </dl>
          {/each}
        {:else}
          <p style="color: var(--text-secondary); margin-top: 0.5rem">No GPU detected — using CPU inference.</p>
        {/if}
      {:else}
        <p style="color: var(--text-secondary)">Detecting hardware...</p>
      {/if}
    </div>
  </section>

  <section>
    <h2>Configuration</h2>
    <div class="card">
      <p style="color: var(--text-secondary)">
        Edit <code>~/.ifran/ifran.toml</code> to configure backends, storage paths, and training settings.
      </p>
    </div>
  </section>
</div>

<style>
  .settings-page {
    display: flex;
    flex-direction: column;
    gap: 1.5rem;
  }

  section {
    display: flex;
    flex-direction: column;
    gap: 0.75rem;
  }

  dl {
    display: grid;
    grid-template-columns: 120px 1fr;
    gap: 0.25rem 1rem;
    font-size: 0.9rem;
  }

  dt {
    color: var(--text-secondary);
    font-weight: 500;
  }

  dd {
    font-family: var(--mono);
    font-size: 0.85rem;
  }
</style>
