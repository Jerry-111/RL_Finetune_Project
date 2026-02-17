#include <Python.h>
#include "drivenet.h"
#define Env Drive
#define MY_SHARED
#define MY_PUT
static PyObject *vec_policy_init(PyObject *self, PyObject *args);
static PyObject *vec_policy_step(PyObject *self, PyObject *args);
static PyObject *vec_policy_close(PyObject *self, PyObject *args);
static PyObject *vec_get_global_agent_meta(PyObject *self, PyObject *args);
#define MY_METHODS                                                                                                    \
    {"vec_policy_init", vec_policy_init, METH_VARARGS, "Initialize native Drive policy runner for a vec env"},      \
        {"vec_policy_step", vec_policy_step, METH_VARARGS, "Run native forward()+c_step() for each env in vec"},    \
        {"vec_policy_close", vec_policy_close, METH_VARARGS, "Free native Drive policy runner for a vec env"},       \
        {"vec_get_global_agent_meta", vec_get_global_agent_meta, METH_VARARGS,                                        \
         "Get per-agent meta for active slots (entity_type, respawn_count)"}
#include "../env_binding.h"

static int my_put(Env *env, PyObject *args, PyObject *kwargs) {
    PyObject *obs = PyDict_GetItemString(kwargs, "observations");
    if (!PyObject_TypeCheck(obs, &PyArray_Type)) {
        PyErr_SetString(PyExc_TypeError, "Observations must be a NumPy array");
        return 1;
    }
    PyArrayObject *observations = (PyArrayObject *)obs;
    if (!PyArray_ISCONTIGUOUS(observations)) {
        PyErr_SetString(PyExc_ValueError, "Observations must be contiguous");
        return 1;
    }
    env->observations = PyArray_DATA(observations);

    PyObject *act = PyDict_GetItemString(kwargs, "actions");
    if (!PyObject_TypeCheck(act, &PyArray_Type)) {
        PyErr_SetString(PyExc_TypeError, "Actions must be a NumPy array");
        return 1;
    }
    PyArrayObject *actions = (PyArrayObject *)act;
    if (!PyArray_ISCONTIGUOUS(actions)) {
        PyErr_SetString(PyExc_ValueError, "Actions must be contiguous");
        return 1;
    }
    env->actions = PyArray_DATA(actions);
    if (PyArray_ITEMSIZE(actions) == sizeof(double)) {
        PyErr_SetString(PyExc_ValueError, "Action tensor passed as float64 (pass np.float32 buffer)");
        return 1;
    }

    PyObject *rew = PyDict_GetItemString(kwargs, "rewards");
    if (!PyObject_TypeCheck(rew, &PyArray_Type)) {
        PyErr_SetString(PyExc_TypeError, "Rewards must be a NumPy array");
        return 1;
    }
    PyArrayObject *rewards = (PyArrayObject *)rew;
    if (!PyArray_ISCONTIGUOUS(rewards)) {
        PyErr_SetString(PyExc_ValueError, "Rewards must be contiguous");
        return 1;
    }
    if (PyArray_NDIM(rewards) != 1) {
        PyErr_SetString(PyExc_ValueError, "Rewards must be 1D");
        return 1;
    }
    env->rewards = PyArray_DATA(rewards);

    PyObject *term = PyDict_GetItemString(kwargs, "terminals");
    if (!PyObject_TypeCheck(term, &PyArray_Type)) {
        PyErr_SetString(PyExc_TypeError, "Terminals must be a NumPy array");
        return 1;
    }
    PyArrayObject *terminals = (PyArrayObject *)term;
    if (!PyArray_ISCONTIGUOUS(terminals)) {
        PyErr_SetString(PyExc_ValueError, "Terminals must be contiguous");
        return 1;
    }
    if (PyArray_NDIM(terminals) != 1) {
        PyErr_SetString(PyExc_ValueError, "Terminals must be 1D");
        return 1;
    }
    env->terminals = PyArray_DATA(terminals);
    return 0;
}

static PyObject *my_shared(PyObject *self, PyObject *args, PyObject *kwargs) {
    char *map_dir = unpack_str(kwargs, "map_dir");
    int num_agents = unpack(kwargs, "num_agents");
    int num_maps = unpack(kwargs, "num_maps");
    int init_mode = unpack(kwargs, "init_mode");
    int control_mode = unpack(kwargs, "control_mode");
    int init_steps = unpack(kwargs, "init_steps");
    int goal_behavior = unpack(kwargs, "goal_behavior");
    float goal_target_distance = unpack(kwargs, "goal_target_distance");

    clock_gettime(CLOCK_REALTIME, &ts);
    srand(ts.tv_nsec); // Always use random sampling with replacement

    int total_agent_count = 0;
    int env_count = 0;

    int max_envs = num_agents;

    int maps_checked = 0;
    PyObject *agent_offsets = PyList_New(max_envs + 1);
    PyObject *map_ids = PyList_New(max_envs);

    // Getting env count
    while (total_agent_count < num_agents && env_count < max_envs) {
        char map_file[512];

        // Always sample randomly with replacement
        int map_id = rand() % num_maps;

        // printf("Sampling map_id: %d\n", map_id);

        Drive *env = calloc(1, sizeof(Drive));
        env->init_mode = init_mode;
        env->control_mode = control_mode;
        env->init_steps = init_steps;
        env->goal_behavior = goal_behavior;
        env->goal_target_distance = goal_target_distance;
        snprintf(map_file, sizeof(map_file), "%s/map_%03d.bin", map_dir, map_id);
        env->entities = load_map_binary(map_file, env);
        set_active_agents(env);

        // Skip map if it doesn't contain any controllable agents
        if (env->active_agent_count == 0) {
            maps_checked++;

            // Safeguard: if we've checked all available maps and found no active agents, raise an error
            if (maps_checked >= num_maps) {
                for (int j = 0; j < env->num_entities; j++) {
                    free_entity(&env->entities[j]);
                }
                free(env->entities);
                free(env->active_agent_indices);
                free(env->static_agent_indices);
                free(env->expert_static_agent_indices);
                free(env);
                Py_DECREF(agent_offsets);
                Py_DECREF(map_ids);
                char error_msg[256];
                sprintf(error_msg, "No controllable agents found in any of the %d available maps", num_maps);
                PyErr_SetString(PyExc_ValueError, error_msg);
                return NULL;
            }

            for (int j = 0; j < env->num_entities; j++) {
                free_entity(&env->entities[j]);
            }
            free(env->entities);
            free(env->active_agent_indices);
            free(env->static_agent_indices);
            free(env->expert_static_agent_indices);
            free(env);
            continue;
        }

        // Store map_id
        PyObject *map_id_obj = PyLong_FromLong(map_id);
        PyList_SetItem(map_ids, env_count, map_id_obj);
        // Store agent offset
        PyObject *offset = PyLong_FromLong(total_agent_count);
        PyList_SetItem(agent_offsets, env_count, offset);
        total_agent_count += env->active_agent_count;
        env_count++;
        for (int j = 0; j < env->num_entities; j++) {
            free_entity(&env->entities[j]);
        }
        free(env->entities);
        free(env->active_agent_indices);
        free(env->static_agent_indices);
        free(env->expert_static_agent_indices);
        free(env);
    }

    if (total_agent_count >= num_agents) {
        total_agent_count = num_agents;
    }

    PyObject *final_total_agent_count = PyLong_FromLong(total_agent_count);
    PyList_SetItem(agent_offsets, env_count, final_total_agent_count);
    PyObject *final_env_count = PyLong_FromLong(env_count);

    // resize lists
    PyObject *resized_agent_offsets = PyList_GetSlice(agent_offsets, 0, env_count + 1);
    PyObject *resized_map_ids = PyList_GetSlice(map_ids, 0, env_count);
    PyObject *tuple = PyTuple_New(3);
    PyTuple_SetItem(tuple, 0, resized_agent_offsets);
    PyTuple_SetItem(tuple, 1, resized_map_ids);
    PyTuple_SetItem(tuple, 2, final_env_count);
    return tuple;
}

static int my_init(Env *env, PyObject *args, PyObject *kwargs) {
    env->human_agent_idx = unpack(kwargs, "human_agent_idx");
    env->ini_file = unpack_str(kwargs, "ini_file");
    env_init_config conf = {0};
    if (ini_parse(env->ini_file, handler, &conf) < 0) {
        printf("Error while loading %s", env->ini_file);
    }
    if (kwargs && PyDict_GetItemString(kwargs, "episode_length")) {
        conf.episode_length = (int)unpack(kwargs, "episode_length");
    }
    if (conf.episode_length <= 0) {
        PyErr_SetString(PyExc_ValueError, "episode_length must be > 0 (set in INI or kwargs)");
        return -1;
    }
    env->action_type = conf.action_type;
    env->dynamics_model = conf.dynamics_model;
    env->reward_vehicle_collision = conf.reward_vehicle_collision;
    env->reward_offroad_collision = conf.reward_offroad_collision;
    env->reward_goal = conf.reward_goal;
    env->reward_goal_post_respawn = conf.reward_goal_post_respawn;
    env->episode_length = conf.episode_length;
    env->termination_mode = conf.termination_mode;
    env->collision_behavior = conf.collision_behavior;
    env->offroad_behavior = conf.offroad_behavior;
    env->max_controlled_agents = unpack(kwargs, "max_controlled_agents");
    env->dt = conf.dt;
    env->init_mode = (int)unpack(kwargs, "init_mode");
    env->control_mode = (int)unpack(kwargs, "control_mode");
    env->goal_behavior = (int)unpack(kwargs, "goal_behavior");
    env->goal_target_distance = (float)unpack(kwargs, "goal_target_distance");
    env->goal_radius = (float)unpack(kwargs, "goal_radius");
    env->goal_speed = (float)unpack(kwargs, "goal_speed");
    char *map_dir = unpack_str(kwargs, "map_dir");
    int map_id = unpack(kwargs, "map_id");
    int max_agents = unpack(kwargs, "max_agents");
    int init_steps = unpack(kwargs, "init_steps");
    char map_file[512];
    snprintf(map_file, sizeof(map_file), "%s/map_%03d.bin", map_dir, map_id);
    env->num_agents = max_agents;
    env->map_name = strdup(map_file);
    env->init_steps = init_steps;
    env->timestep = init_steps;
    init(env);
    return 0;
}

static int my_log(PyObject *dict, Log *log) {
    assign_to_dict(dict, "n", log->n);
    assign_to_dict(dict, "score", log->score);
    assign_to_dict(dict, "offroad_rate", log->offroad_rate);
    assign_to_dict(dict, "collision_rate", log->collision_rate);
    assign_to_dict(dict, "episode_length", log->episode_length);
    assign_to_dict(dict, "episode_return", log->episode_return);
    assign_to_dict(dict, "dnf_rate", log->dnf_rate);
    assign_to_dict(dict, "completion_rate", log->completion_rate);
    assign_to_dict(dict, "lane_alignment_rate", log->lane_alignment_rate);
    assign_to_dict(dict, "offroad_per_agent", log->offroad_per_agent);
    assign_to_dict(dict, "collisions_per_agent", log->collisions_per_agent);
    assign_to_dict(dict, "goals_sampled_this_episode", log->goals_sampled_this_episode);
    assign_to_dict(dict, "goals_reached_this_episode", log->goals_reached_this_episode);
    assign_to_dict(dict, "speed_at_goal", log->speed_at_goal);
    // assign_to_dict(dict, "avg_displacement_error", log->avg_displacement_error);
    return 0;
}

static PyObject *vec_get_global_agent_meta(PyObject *self, PyObject *args) {
    if (PyTuple_Size(args) != 3) {
        PyErr_SetString(PyExc_TypeError, "vec_get_global_agent_meta requires 3 arguments");
        return NULL;
    }

    VecEnv *vec = unpack_vecenv(args);
    if (!vec) {
        return NULL;
    }

    PyObject *type_arr = PyTuple_GetItem(args, 1);
    PyObject *respawn_arr = PyTuple_GetItem(args, 2);
    if (!PyArray_Check(type_arr) || !PyArray_Check(respawn_arr)) {
        PyErr_SetString(PyExc_TypeError, "All output arrays must be NumPy arrays");
        return NULL;
    }

    PyArrayObject *type_array = (PyArrayObject *)type_arr;
    PyArrayObject *respawn_array = (PyArrayObject *)respawn_arr;
    if (!PyArray_ISCONTIGUOUS(type_array) || !PyArray_ISCONTIGUOUS(respawn_array)) {
        PyErr_SetString(PyExc_ValueError, "Output arrays must be contiguous");
        return NULL;
    }

    int *type_base = (int *)PyArray_DATA(type_array);
    int *respawn_base = (int *)PyArray_DATA(respawn_array);

    int offset = 0;
    for (int i = 0; i < vec->num_envs; i++) {
        Drive *drive = (Drive *)vec->envs[i];
        for (int j = 0; j < drive->active_agent_count; j++) {
            int entity_idx = drive->active_agent_indices[j];
            type_base[offset + j] = drive->entities[entity_idx].type;
            respawn_base[offset + j] = drive->entities[entity_idx].respawn_count;
        }
        offset += drive->active_agent_count;
    }
    Py_RETURN_NONE;
}

typedef struct VecPolicyRuntime VecPolicyRuntime;
struct VecPolicyRuntime {
    VecEnv *vec;
    Weights *weights;
    DriveNet **nets;
    int num_envs;
    char *policy_path;
    VecPolicyRuntime *next;
};

static VecPolicyRuntime *g_policy_runtimes = NULL;

static VecPolicyRuntime *find_runtime(VecEnv *vec) {
    VecPolicyRuntime *cur = g_policy_runtimes;
    while (cur != NULL) {
        if (cur->vec == vec) {
            return cur;
        }
        cur = cur->next;
    }
    return NULL;
}

static void free_runtime_contents(VecPolicyRuntime *rt) {
    if (rt == NULL) {
        return;
    }
    if (rt->nets != NULL) {
        for (int i = 0; i < rt->num_envs; i++) {
            if (rt->nets[i] != NULL) {
                free_drivenet(rt->nets[i]);
            }
        }
        free(rt->nets);
        rt->nets = NULL;
    }
    if (rt->weights != NULL) {
        free(rt->weights);
        rt->weights = NULL;
    }
    if (rt->policy_path != NULL) {
        free(rt->policy_path);
        rt->policy_path = NULL;
    }
    rt->num_envs = 0;
}

static void remove_runtime(VecEnv *vec) {
    VecPolicyRuntime *prev = NULL;
    VecPolicyRuntime *cur = g_policy_runtimes;
    while (cur != NULL) {
        if (cur->vec == vec) {
            if (prev == NULL) {
                g_policy_runtimes = cur->next;
            } else {
                prev->next = cur->next;
            }
            free_runtime_contents(cur);
            free(cur);
            return;
        }
        prev = cur;
        cur = cur->next;
    }
}

static PyObject *vec_policy_init(PyObject *self, PyObject *args) {
    if (PyTuple_Size(args) != 2) {
        PyErr_SetString(PyExc_TypeError, "vec_policy_init requires 2 arguments: vec_handle, policy_path");
        return NULL;
    }

    VecEnv *vec = unpack_vecenv(args);
    if (!vec) {
        return NULL;
    }

    PyObject *path_obj = PyTuple_GetItem(args, 1);
    if (!PyUnicode_Check(path_obj)) {
        PyErr_SetString(PyExc_TypeError, "policy_path must be a string");
        return NULL;
    }
    const char *policy_path = PyUnicode_AsUTF8(path_obj);
    if (policy_path == NULL) {
        return NULL;
    }

    FILE *policy_file = fopen(policy_path, "rb");
    if (policy_file == NULL) {
        char msg[512];
        snprintf(msg, sizeof(msg), "Policy file not found: %s", policy_path);
        PyErr_SetString(PyExc_FileNotFoundError, msg);
        return NULL;
    }
    fclose(policy_file);

    remove_runtime(vec);

    VecPolicyRuntime *rt = (VecPolicyRuntime *)calloc(1, sizeof(VecPolicyRuntime));
    if (rt == NULL) {
        PyErr_SetString(PyExc_MemoryError, "Failed to allocate VecPolicyRuntime");
        return NULL;
    }
    rt->vec = vec;
    rt->num_envs = vec->num_envs;
    rt->policy_path = strdup(policy_path);
    if (rt->policy_path == NULL) {
        free(rt);
        PyErr_SetString(PyExc_MemoryError, "Failed to allocate policy path");
        return NULL;
    }

    rt->weights = load_weights((char *)policy_path);
    if (rt->weights == NULL) {
        free_runtime_contents(rt);
        free(rt);
        PyErr_SetString(PyExc_RuntimeError, "load_weights failed");
        return NULL;
    }

    rt->nets = (DriveNet **)calloc(rt->num_envs, sizeof(DriveNet *));
    if (rt->nets == NULL) {
        free_runtime_contents(rt);
        free(rt);
        PyErr_SetString(PyExc_MemoryError, "Failed to allocate DriveNet array");
        return NULL;
    }

    for (int i = 0; i < rt->num_envs; i++) {
        Drive *drive = (Drive *)vec->envs[i];
        rt->weights->idx = 0;
        rt->nets[i] = init_drivenet(rt->weights, drive->active_agent_count, drive->dynamics_model);
        if (rt->nets[i] == NULL) {
            free_runtime_contents(rt);
            free(rt);
            PyErr_SetString(PyExc_RuntimeError, "init_drivenet failed");
            return NULL;
        }
    }

    rt->next = g_policy_runtimes;
    g_policy_runtimes = rt;
    Py_RETURN_NONE;
}

static PyObject *vec_policy_step(PyObject *self, PyObject *args) {
    if (PyTuple_Size(args) != 1) {
        PyErr_SetString(PyExc_TypeError, "vec_policy_step requires 1 argument: vec_handle");
        return NULL;
    }

    VecEnv *vec = unpack_vecenv(args);
    if (!vec) {
        return NULL;
    }

    VecPolicyRuntime *rt = find_runtime(vec);
    if (rt == NULL) {
        PyErr_SetString(PyExc_RuntimeError, "Policy runtime not initialized. Call vec_policy_init first.");
        return NULL;
    }
    if (rt->num_envs != vec->num_envs) {
        PyErr_SetString(PyExc_RuntimeError, "vec size changed; reinitialize policy runtime");
        return NULL;
    }

    for (int i = 0; i < vec->num_envs; i++) {
        Drive *drive = (Drive *)vec->envs[i];
        DriveNet *net = rt->nets[i];
        if (net == NULL) {
            PyErr_SetString(PyExc_RuntimeError, "Missing DriveNet for env index");
            return NULL;
        }
        if (net->num_agents != drive->active_agent_count) {
            PyErr_SetString(PyExc_RuntimeError, "active_agent_count changed; reinitialize policy runtime");
            return NULL;
        }
        forward(net, drive->observations, (int *)drive->actions);
        c_step(drive);
    }
    Py_RETURN_NONE;
}

static PyObject *vec_policy_close(PyObject *self, PyObject *args) {
    if (PyTuple_Size(args) != 1) {
        PyErr_SetString(PyExc_TypeError, "vec_policy_close requires 1 argument: vec_handle");
        return NULL;
    }

    VecEnv *vec = unpack_vecenv(args);
    if (!vec) {
        return NULL;
    }
    remove_runtime(vec);
    Py_RETURN_NONE;
}
