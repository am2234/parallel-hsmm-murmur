# viterbi_hmm.pyx is © 2022, University of Cambridge
#
# viterbi_hmm.pyx is published and distributed under the GAP Available Source License v1.0 (ASL).
#
# viterbi_hmm.pyx is is distributed in the hope that it will be useful for non-commercial academic 
# research, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY 
# or FITNESS FOR A PARTICULAR PURPOSE. See the ASL for more details. 
#
# You should have received a copy of the ASL along with this program; if not, write to 
# am2234 (at) cam (dot) ac (dot) uk. 
import numpy as np
from libc.math cimport log, INFINITY
from libc.float cimport FLT_MIN
cimport cython

DTYPE = np.intc

#@cython.boundscheck(False)  # Deactivate bounds checking
@cython.wraparound(False)   # Deactivate negative indexing.
def hsmm_viterbi(float [:,:] posteriors, float [:,:] durations, Py_ssize_t max_duration,
                 float [:,:] A):
    # durations[d,s] gives the p_s(d+1), probability of d+1 consecutive observations in state s
    # e.g. duration[0,s] gives probability of only staying in state s for one sample then moving

    cdef Py_ssize_t T = posteriors.shape[0]
    cdef Py_ssize_t N = posteriors.shape[1]

    psi_np = np.empty((T+max_duration, N),dtype=np.float32)
    psi_np[:] = -np.inf
    psi_np[0, :] = np.log(posteriors[0, :])  # Uniform priors so not multiplied in here

    psi_arg_np = np.empty((T+max_duration, N), dtype=np.intc)
    psi_duration_np = np.empty((T+max_duration, N), dtype=np.intc)

    cdef float [:,::1] delta = psi_np
    cdef int [:,::1] psi = psi_arg_np
    cdef int [:,::1] psi_duration = psi_duration_np

    cdef float delta_this_duration, product_observation_probs, temp_delta, delta_max
    cdef Py_ssize_t t,s,d,i,i_max,start_t,end_t

    for t in range(1, T + max_duration):
        for s in range(N):
            for d in range(1, max_duration+1): # <=
                start_t = max(0, min(t - d, T - 1))  # clamp (t-d) to [0, T-2] range
                end_t = min(t, T)

                # Calculate maximum probability up to this point, including transitioning to this state
                delta_max = -1 * INFINITY
                i_max = -1
                for i in range(N):
                    temp_delta = delta[start_t,i] + log(A[i][s] + FLT_MIN)
                    if temp_delta > delta_max:
                        delta_max = temp_delta
                        i_max = i

                # Calculate probability of observing the same state for this duration
                product_observation_probs = 0
                for i in range(start_t, end_t): # < not <= according to Schmidt
                    product_observation_probs += log(posteriors[i,s] + FLT_MIN)

                # Add previous two terms with explicit probability of this state duration
                delta_this_duration = delta_max + product_observation_probs + log(durations[d-1,s] + FLT_MIN)

                # Save best result, including the state we transitioned from and the duration spent in new state
                if delta_this_duration > delta[t, s]:
                    delta[t,s] = delta_this_duration
                    psi[t,s] = i_max
                    psi_duration[t,s] = d

    cdef int current_state = -1
    cdef int end_time  = -1
    cdef float max_delta_after = -1 * INFINITY

    # Find most likely state after end of sequence (Springer extended Viterbi)
    for t in range(T, T+max_duration):
        for s in range(N):
            if delta[t,s] > max_delta_after:
               current_state = s
               end_time = t
               max_delta_after = delta[t,s]

    states_np = -np.ones(T+max_duration, dtype=np.int32)
    cdef int[:] states = states_np
    states[end_time] = current_state
    t = end_time

    while t > 0:
        d = psi_duration[t, current_state]
        for i in range(max(0, t-d), t):
            states[i] = current_state

        t = max(0,t-d)

        current_state = psi[t, current_state]

    return states_np[:T]
