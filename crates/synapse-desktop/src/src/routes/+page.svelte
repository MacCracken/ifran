<script lang="ts">
  import { onMount } from "svelte";
  import { refreshStatus, status } from "$lib/stores/system";
  import { refreshHardware, hardware } from "$lib/stores/system";
  import { refreshModels, models } from "$lib/stores/models";

  onMount(() => {
    refreshStatus();
    refreshHardware();
    refreshModels();
  });
</script>

<div class="dashboard">
  <h1>Dashboard</h1>

  <div class="grid">
    <div class="card">
      <h3>Models</h3>
      <p class="stat">{$models.length}</p>
      <p class="label">in catalog</p>
    </div>

    <div class="card">
      <h3>Status</h3>
      {#if $status}
        <p class="stat">{$status.loaded_models}</p>
        <p class="label">loaded models</p>
      {:else}
        <p class="label">Connecting...</p>
      {/if}
    </div>

    <div class="card">
      <h3>Hardware</h3>
      {#if $hardware}
        {#if $hardware.has_gpu}
          <p class="stat">{$hardware.gpus.length}</p>
          <p class="label">GPU{$hardware.gpus.length > 1 ? 's' : ''} — {$hardware.total_gpu_memory_mb} MB</p>
        {:else}
          <p class="stat">CPU</p>
          <p class="label">{$hardware.cpu.cores} cores / {$hardware.cpu.memory_total_mb} MB</p>
        {/if}
      {:else}
        <p class="label">Detecting...</p>
      {/if}
    </div>

    <div class="card">
      <h3>Version</h3>
      <p class="stat">{$status?.version ?? '—'}</p>
      <p class="label">synapse</p>
    </div>
  </div>
</div>

<style>
  .dashboard {
    display: flex;
    flex-direction: column;
    gap: 1.5rem;
  }

  .grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
    gap: 1rem;
  }

  .stat {
    font-size: 2rem;
    font-weight: 700;
    color: var(--accent);
    margin: 0.5rem 0 0.25rem;
  }

  .label {
    color: var(--text-secondary);
    font-size: 0.85rem;
  }
</style>
