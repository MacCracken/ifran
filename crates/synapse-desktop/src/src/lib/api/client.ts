import { invoke } from "@tauri-apps/api/core";

export interface Model {
  id: string;
  name: string;
  repo_id: string | null;
  format: string;
  quant: string;
  size_bytes: number;
  parameter_count: number | null;
  architecture: string | null;
  local_path: string;
  pulled_at: string;
}

export interface SystemStatus {
  version: string;
  loaded_models: number;
  registered_backends: string[];
  hardware: HardwareInfo;
}

export interface HardwareInfo {
  cpu: {
    model: string;
    cores: number;
    threads: number;
    memory_total_mb: number;
    memory_available_mb: number;
  };
  gpus: GpuInfo[];
  has_gpu: boolean;
  total_gpu_memory_mb: number;
}

export interface GpuInfo {
  index: number;
  name: string;
  memory_total_mb: number;
  memory_free_mb: number;
}

export interface TrainingJob {
  id: string;
  status: string;
  current_step: number;
  total_steps: number;
  current_epoch: number;
  current_loss: number | null;
  progress_percent: number;
  error: string | null;
  created_at: string;
  started_at: string | null;
  completed_at: string | null;
}

export interface ChatResponse {
  choices: { message: { content: string } }[];
}

// Models
export async function listModels(): Promise<{ data: Model[] }> {
  return invoke("list_models");
}

export async function getModel(id: string): Promise<Model> {
  return invoke("get_model", { id });
}

export async function deleteModel(id: string): Promise<void> {
  return invoke("delete_model", { id });
}

export async function pullModel(
  repoId: string,
  quant?: string,
): Promise<unknown> {
  return invoke("pull_model", { repoId, quant });
}

// Chat
export async function sendMessage(
  model: string,
  prompt: string,
  systemPrompt?: string,
  maxTokens?: number,
  temperature?: number,
): Promise<ChatResponse> {
  return invoke("send_message", {
    model,
    prompt,
    systemPrompt,
    maxTokens,
    temperature,
  });
}

// System
export async function getStatus(): Promise<SystemStatus> {
  return invoke("get_status");
}

export async function getHardware(): Promise<HardwareInfo> {
  return invoke("get_hardware");
}

// Training
export async function listJobs(): Promise<TrainingJob[]> {
  return invoke("list_jobs");
}

export async function createJob(config: unknown): Promise<TrainingJob> {
  return invoke("create_job", { config });
}

export async function cancelJob(id: string): Promise<TrainingJob> {
  return invoke("cancel_job", { id });
}
