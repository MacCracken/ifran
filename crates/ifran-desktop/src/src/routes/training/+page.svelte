<script lang="ts">
  import { onMount } from "svelte";
  import { listJobs, cancelJob, type TrainingJob } from "$lib/api/client";

  let jobs: TrainingJob[] = $state([]);
  let loading = $state(true);

  onMount(refresh);

  async function refresh() {
    loading = true;
    try {
      jobs = await listJobs();
    } catch {
      jobs = [];
    } finally {
      loading = false;
    }
  }

  async function handleCancel(id: string) {
    if (confirm("Cancel this training job?")) {
      await cancelJob(id);
      await refresh();
    }
  }

  function statusBadge(status: string): string {
    switch (status) {
      case "running": return "badge-info";
      case "completed": return "badge-success";
      case "failed": return "badge-error";
      case "cancelled": return "badge-warning";
      default: return "";
    }
  }
</script>

<div class="training-page">
  <div class="header">
    <h1>Training Jobs</h1>
    <button class="btn" onclick={refresh}>Refresh</button>
  </div>

  {#if loading}
    <p style="color: var(--text-secondary)">Loading...</p>
  {:else if jobs.length === 0}
    <div class="card">
      <p>No training jobs. Create one with <code>ifran train</code> or the API.</p>
    </div>
  {:else}
    <div class="job-list">
      {#each jobs as job}
        <div class="card job-card">
          <div class="job-info">
            <div class="job-header">
              <code>{job.id.slice(0, 8)}</code>
              <span class="badge {statusBadge(job.status)}">{job.status}</span>
            </div>

            {#if job.status === "running"}
              <div class="progress-bar">
                <div class="progress-fill" style="width: {job.progress_percent}%"></div>
              </div>
              <div class="progress-meta">
                <span>Step {job.current_step}/{job.total_steps}</span>
                <span>Epoch {job.current_epoch.toFixed(1)}</span>
                {#if job.current_loss}
                  <span>Loss: {job.current_loss.toFixed(4)}</span>
                {/if}
                <span>{job.progress_percent.toFixed(1)}%</span>
              </div>
            {/if}

            {#if job.error}
              <p style="color: var(--error); font-size: 0.85rem">{job.error}</p>
            {/if}

            <p class="meta">Created: {new Date(job.created_at).toLocaleString()}</p>
          </div>

          <div class="job-actions">
            {#if job.status === "running" || job.status === "queued"}
              <button class="btn btn-danger" onclick={() => handleCancel(job.id)}>
                Cancel
              </button>
            {/if}
          </div>
        </div>
      {/each}
    </div>
  {/if}
</div>

<style>
  .training-page {
    display: flex;
    flex-direction: column;
    gap: 1rem;
  }

  .header {
    display: flex;
    justify-content: space-between;
    align-items: center;
  }

  .job-list {
    display: flex;
    flex-direction: column;
    gap: 0.75rem;
  }

  .job-card {
    display: flex;
    justify-content: space-between;
    align-items: flex-start;
  }

  .job-info {
    flex: 1;
    display: flex;
    flex-direction: column;
    gap: 0.5rem;
  }

  .job-header {
    display: flex;
    align-items: center;
    gap: 0.75rem;
  }

  .progress-bar {
    width: 100%;
    height: 6px;
    background: var(--bg-tertiary);
    border-radius: 3px;
    overflow: hidden;
  }

  .progress-fill {
    height: 100%;
    background: var(--accent);
    border-radius: 3px;
    transition: width 0.3s;
  }

  .progress-meta {
    display: flex;
    gap: 1rem;
    font-size: 0.8rem;
    color: var(--text-secondary);
    font-family: var(--mono);
  }

  .meta {
    font-size: 0.8rem;
    color: var(--text-secondary);
  }
</style>
