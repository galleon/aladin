import axios from 'axios';

const API_PATH = import.meta.env.VITE_API_URL || '/api';

export const apiClient = axios.create({
    baseURL: API_PATH,
    headers: {
        'Content-Type': 'application/json',
    },
});

export interface Tenant {
    id: number;
    name: string;
    namespace: string;
    status: string;
    helm_release_name: string;
    created_at: string;
    updated_at: string;
}

export interface ClusterCondition {
    type: string;
    status: string;
    reason?: string;
    message?: string;
    lastTransitionTime?: string;
}

export interface ClusterStatus {
    metadata?: {
        name?: string;
    };
    status?: {
        phase?: string;
        conditions?: ClusterCondition[];
    };
}

export interface TenantStatusResponse {
    tenant: Tenant;
    namespace: string;
    cluster: ClusterStatus | null;
    helm: any;
    deployment: any;
}

export interface CreateTenantRequest {
    name: string;
    namespace: string;
    helmValues?: Record<string, any>;
}

export interface TokenUsage {
    id: number;
    tenant_id: number;
    timestamp: string;
    input_tokens: number;
    output_tokens: number;
    total_tokens: number;
    model?: string;
    endpoint?: string;
}

export const tenantsApi = {
    getAll: () => apiClient.get<Tenant[]>('/tenants'),
    getById: (id: number) => apiClient.get<Tenant>(`/tenants/${id}`),
    create: (data: CreateTenantRequest) => apiClient.post<Tenant>('/tenants', data),
    delete: (id: number) => apiClient.delete(`/tenants/${id}`),
    getStatus: (id: number) => apiClient.get<TenantStatusResponse>(`/tenants/${id}/status`),
};

export const monitoringApi = {
    getTenantMonitoring: (id: number) => apiClient.get(`/monitoring/tenant/${id}`),
    getAllTenantsMonitoring: () => apiClient.get('/monitoring/tenants'),
};

export const billingApi = {
    recordUsage: (data: {
        tenantId: number;
        inputTokens: number;
        outputTokens: number;
        model?: string;
        endpoint?: string;
    }) => apiClient.post('/billing/usage', data),
    getUsage: (tenantId: number, startDate?: string, endDate?: string) => {
        const params = new URLSearchParams();
        if (startDate) params.append('startDate', startDate);
        if (endDate) params.append('endDate', endDate);
        return apiClient.get<TokenUsage[]>(`/billing/usage/tenant/${tenantId}?${params}`);
    },
    getSummary: (tenantId: number) => apiClient.get(`/billing/usage/tenant/${tenantId}/summary`),
    getByModel: (tenantId: number) => apiClient.get(`/billing/usage/tenant/${tenantId}/by-model`),
};

export interface Application {
    id: string;
    name: string;
    description: string;
    version: string;
    icon?: string;
    category?: string;
}

export interface DeployedApplication {
    id: number;
    tenant_id: number;
    application_id: string;
    release_name: string;
    namespace: string;
    status: string;
    helm_values: Record<string, any>;
    created_at: string;
    updated_at: string;
}

export interface DeployApplicationRequest {
    application_id: string;
    release_name?: string;
    values?: Record<string, any>;
}

export const applicationsApi = {
    getAll: () => apiClient.get<Application[]>('/applications'),
    getById: (id: string) => apiClient.get<Application>(`/applications/${id}`),
    getDeployments: (tenantId: number) => apiClient.get<DeployedApplication[]>(`/applications/tenants/${tenantId}/deployments`),
    deploy: (tenantId: number, data: DeployApplicationRequest) => apiClient.post<DeployedApplication>(`/applications/tenants/${tenantId}/deploy`, data),
    uninstall: (tenantId: number, deploymentId: number) => apiClient.delete(`/applications/tenants/${tenantId}/deployments/${deploymentId}`),
};

export interface Task {
    id: number;
    task_id: string;
    task_type: string;
    status: 'pending' | 'running' | 'success' | 'failed';
    result?: Record<string, any>;
    error_message?: string;
    progress: number;
    created_at: string;
    updated_at: string;
    completed_at?: string;
    tenant_id?: number;
    deployment_id?: number;
}

export interface TaskStatusUpdate {
    type: string;
    task_id: string;
    status: string;
    progress: number;
    result?: Record<string, any>;
    error_message?: string;
}

export const tasksApi = {
    getAll: (params?: { status?: string; task_type?: string; tenant_id?: number; limit?: number }) => {
        const queryParams = new URLSearchParams();
        if (params?.status) queryParams.append('status', params.status);
        if (params?.task_type) queryParams.append('task_type', params.task_type);
        if (params?.tenant_id) queryParams.append('tenant_id', params.tenant_id.toString());
        if (params?.limit) queryParams.append('limit', params.limit.toString());
        return apiClient.get<Task[]>(`/tasks?${queryParams}`);
    },
    getById: (taskId: string) => apiClient.get<Task>(`/tasks/${taskId}`),
};

