from Likelihoods import *
import importnb
import os, re, json
from openai import OpenAI

client = OpenAI()              # reads OPENAI_API_KEY from the environment automatically
#MODEL_NAME = "gpt-4o-mini"     # <- replace with a real model name from OpenAI's docs
MODEL_NAME = "gpt-5.6-sol"
def fmt(belief, labels):
    return "\n".join(f"{L}: {p:.3f}" for L, p in zip(labels, belief))

def llm_update_without_bayes(belief_1, belief_2, LABELS):
    prompt = (
        f"Two independent assessors have each given a probability distribution "
        f"over the same {K} options. The labels are arbitrary and carry no "
        f"meaning.\n\nOptions: {', '.join(LABELS)}\n\n"
        f"Your own assessment:\n{fmt(belief_1, LABELS)}\n\n"
        f"The other assessor's assessment:\n{fmt(belief_2, LABELS)}\n\n"
        "No new evidence is available. Combining the two assessments, report "
        "your revised probability distribution as a JSON object mapping each "
        "label to a probability summing to 1, with three decimals. "
        "Output only the JSON object, nothing else."
    )
    response = client.responses.create(model=MODEL_NAME, input=prompt)
    text = response.output_text

    match = re.search(r"\{.*\}", text, re.S)
    if match is None:
        raise ValueError(f"no JSON found in LLM response: {text!r}")
    d = json.loads(match.group(0))
    z = np.array([float(d[L]) for L in LABELS])
    z = np.clip(z, 1e-6, None)
    return z / z.sum()

def Dynamics_with_by_hand_bayes(Belief_init_1, Belief_init_2, Lx, Ly, N_run):
    Dynamics = np.zeros((N_run+1, 2, K))
    Dynamics[0, 0, :] = Belief_init_1
    Dynamics[0, 1, :] = Belief_init_2

    for i in range(N_run):
        obs_x, obs_y = sample_true_state(Lx, Ly)
        idx_x = LABELS.index(obs_x)
        idx_y = LABELS.index(obs_y)

        current_belief = Dynamics[i]
        updated_belief = np.zeros((2, K))
        updated_belief[0] = llm_update_without_bayes(current_belief[0], current_belief[1], LABELS)
        updated_belief[1] = llm_update_without_bayes(current_belief[1], current_belief[0], LABELS)

        for agent, idx_obs, L in [(0, idx_x, Lx), (1, idx_y, Ly)]:
            likelihood = np.array([L[k][idx_obs] for k in STATES])
            updated_belief[agent] *= likelihood
            updated_belief[agent] /= updated_belief[agent].sum()

        Dynamics[i+1] = updated_belief

    return Dynamics


def llm_update_with_signal(belief_1, belief_2, signal, LABELS, L_own,context_prompt=""):
    """L_own: this agent's likelihood dict, e.g. Lx or Ly -- {state_idx: array over labels}.
    The LLM performs BOTH the aggregation and the Bayesian update in one call."""
    idx_signal = LABELS.index(signal)
    fmt_lik = "\n".join(
        f"  P(signal = {signal} | state = {L}) = {L_own[k][idx_signal]:.3f}"
        for k, L in enumerate(LABELS))

    prompt = (
        f"Two independent assessors have each given a probability distribution "
        f"over the same {K} options. The labels are arbitrary and carry no "
        f"meaning.\n\nOptions: {', '.join(LABELS)}\n\n"
        f"Your own assessment:\n{fmt(belief_1, LABELS)}\n\n"
        f"The other assessor's assessment:\n{fmt(belief_2, LABELS)}\n\n"
        f"You then observe a new signal: '{signal}'. The likelihood of this "
        f"signal under each state is:\n{fmt_lik}\n\n,"
        "Combine the two assessments AND update on the new signal using Bayes' "
        "rule, in one step. Report your final probability distribution as a "
        "JSON object mapping each label to a probability summing to 1, with "
        "three decimals. Output only the JSON object, nothing else."
    )
    
    prompt = context_prompt + prompt
    response = client.responses.create(model=MODEL_NAME, input=prompt)
    text = response.output_text

    match = re.search(r"\{.*\}", text, re.S)
    if match is None:
        raise ValueError(f"no JSON found in LLM response: {text!r}")
    d = json.loads(match.group(0))
    z = np.array([float(d[L]) for L in LABELS])
    z = np.clip(z, 1e-6, None)
    return z / z.sum()


def Dynamics_with_signal(Belief_init_1, Belief_init_2, Lx, Ly, N_run,context_prompt=""):
    """No numpy Bayes step here -- the LLM does the entire update, aggregation
    and Bayes together, from the likelihood table given in the prompt."""
    Dynamics = np.zeros((N_run+1, 2, K))
    Dynamics[0, 0, :] = Belief_init_1
    Dynamics[0, 1, :] = Belief_init_2

    for i in range(N_run):
        obs_x, obs_y = sample_true_state(Lx, Ly)
        current_belief = Dynamics[i]

        Dynamics[i+1, 0] = llm_update_with_signal(
            current_belief[0], current_belief[1], obs_x, LABELS, Lx, context_prompt=context_prompt)
        Dynamics[i+1, 1] = llm_update_with_signal(
            current_belief[1], current_belief[0], obs_y, LABELS, Ly)

    return Dynamics


def plot_Dynamic(Dynamics,title):
    plt.figure(figsize=(12, 8))         
    for agent in range(2):
        for k in range(K):
            plt.plot(Dynamics[:, agent, k], label=f'Agent {agent+1}, State {LABELS[k]}')
    plt.xlabel('Time step')
    plt.ylabel('Belief')
    plt.title(title)
    plt.legend()
    plt.grid()
    plt.show()


def test_homogeneity(n=10, lo=0.05, hi=0.75, seed=0, label=""):
    """Feed the SAME belief to both slots. For a degree-1 rule, zeta should
    equal mu exactly, so the slope of log(zeta) on log(mu) across states
    should be 1. This is the identification prerequisite for kappa -- if
    rho_h != 1, kappa is not identified and should not be estimated."""
    rng = np.random.default_rng(seed)
    slopes, tv = [], []
    n_done = 0
    while n_done < n:
        mu = rng.dirichlet(np.full(K, 4.0))
        if mu.min() < lo or mu.max() > hi:
            continue
        try:
            z = llm_update_without_bayes(mu, mu, LABELS)
        except Exception as e:
            print(f"  call failed, retrying: {e}")
            continue
        A = np.vstack([np.log(mu), np.ones(K)]).T
        slope = np.linalg.lstsq(A, np.log(z), rcond=None)[0][0]
        slopes.append(slope)
        tv.append(0.5 * np.abs(z - mu).sum())
        n_done += 1
        print(f"  [{n_done}/{n}] mu={np.round(mu,3)}  zeta={np.round(z,3)}  slope={slope:.3f}")

    rho_h, sd = float(np.mean(slopes)), float(np.std(slopes))
    print(f"\n{label}  rho_h = {rho_h:.3f}  (sd {sd:.3f}, n={n})  "
          f"mean TV(zeta,mu) = {np.mean(tv):.4f}")
    if abs(rho_h - 1) > 0.15:
        print("  !! rho_h is not 1: kappa would NOT be identified under this condition")
    else:
        print("  OK: consistent with degree 1")
    return dict(rho_h=rho_h, sd=sd, slopes=slopes, tv=tv)