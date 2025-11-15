import numpy as np
import math
import json
from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator

# --- Importuri pentru API ---
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import List, Dict, Any  # Tipuri de date

# --- Inițializare API ---
app = FastAPI(
    title="Grover vs. Classical Search API",
    description="Compară simulările algoritmului Grover (cuantic) cu o căutare liniară (clasică).",
    version="1.1.0"
)

# --- Inițializare Simulator Qiskit (o singură dată) ---
simulator = AerSimulator()


# =======================================================
#  DEFINIȚIILE FUNCȚIILOR QISKIT (CUANTICE)
# =======================================================

def grover_oracle(num_qubits, target_state):
    """Construiește oracolul care marchează starea țintă."""
    qc = QuantumCircuit(num_qubits)
    bits = format(target_state, f"0{num_qubits}b")[::-1]
    for i, b in enumerate(bits):
        if b == "0":
            qc.x(i)
    qc.h(num_qubits - 1)
    qc.mcx(list(range(num_qubits - 1)), num_qubits - 1)
    qc.h(num_qubits - 1)
    for i, b in enumerate(bits):
        if b == "0":
            qc.x(i)
    return qc.to_gate(label="Oracle")


def diffusion_operator(n):
    """Construiește operatorul de difuzie (amplificare)."""
    qc = QuantumCircuit(n)
    qc.h(range(n))
    qc.x(range(n))
    qc.h(n - 1)
    qc.mcx(list(range(n - 1)), n - 1)
    qc.h(n - 1)
    qc.x(range(n))
    qc.h(range(n))
    return qc.to_gate(label="Diffuser")


def run_grover_simulation(num_qubits, target_state, max_iterations, shots=1024):
    """Rulează simularea cuantică completă și returnează datele de evoluție."""
    N_sim_space = 2 ** num_qubits
    evolution_data = []
    oracle = grover_oracle(num_qubits, target_state)
    diffuser = diffusion_operator(num_qubits)

    for i in range(max_iterations + 1):
        qc = QuantumCircuit(num_qubits, num_qubits)
        qc.h(range(num_qubits))
        for _ in range(i):
            qc.append(oracle, qc.qubits)
            qc.append(diffuser, qc.qubits)
        qc.measure(range(num_qubits), range(num_qubits))

        compiled_circuit = transpile(qc, simulator)
        job = simulator.run(compiled_circuit, shots=shots)
        result = job.result()
        counts = result.get_counts(compiled_circuit)

        probabilities_all_states = [0.0] * N_sim_space
        for state_binary, count in counts.items():
            state_int = int(state_binary, 2)
            probabilities_all_states[state_int] = round(count / shots, 6)

        prob_list_of_dicts = [
            {"state_int": idx, "state_binary": format(idx, f"0{num_qubits}b"), "probability": prob}
            for idx, prob in enumerate(probabilities_all_states)
        ]
        evolution_data.append({"iteration": i, "probabilities": prob_list_of_dicts})
    return evolution_data


# =======================================================
#  DEFINIȚIILE ENDPOINT-URILOR API
# =======================================================

# --- Modele de Date (Pydantic) ---

# Modelul de date pentru request (comun pentru ambele endpoint-uri)
class GroverSearchRequest(BaseModel):
    list_size_N: int = Field(..., gt=1, description="Dimensiunea listei de căutare (ex: 10, 100, 1000).")
    target_number: int = Field(..., ge=0, description="Numărul de căutat (ex: 7).")


# Modelul de date pentru response-ul CUANTIC
class GroverSimulationResponse(BaseModel):
    metadata: Dict[str, Any]
    evolution_by_iteration: List[Dict[str, Any]]


# Modelul de date pentru response-ul CLASIC
class ClassicalStep(BaseModel):
    step: int
    checking_number: int
    found: bool


class ClassicalSearchResponse(BaseModel):
    metadata: Dict[str, Any]
    search_log: List[ClassicalStep]


# --- Validare comună ---
def validate_request(request: GroverSearchRequest):
    if request.target_number >= request.list_size_N:
        raise HTTPException(
            status_code=400,
            detail="Numărul țintă (target_number) trebuie să fie mai mic decât dimensiunea listei (list_size_N)."
        )


# --- Endpoint 1: Căutare CUANTICĂ (Grover) ---
@app.post("/search", response_model=GroverSimulationResponse, tags=["Quantum"])
async def search_with_grover(request: GroverSearchRequest):
    """
    Rulează algoritmul lui Grover (Cuantic).

    Calculează numărul de qubiți necesari, simulează evoluția
    probabilităților și returnează rezultatul.
    """
    validate_request(request)

    total_qubits = math.ceil(math.log2(request.list_size_N))
    N_sim_space = 2 ** total_qubits
    optimal_iterations = round((math.pi / 4) * math.sqrt(N_sim_space))
    max_iterations_to_run = optimal_iterations + 3

    try:
        evolution_data = run_grover_simulation(
            num_qubits=total_qubits,
            target_state=request.target_number,
            max_iterations=max_iterations_to_run
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Eroare în simularea Qiskit: {str(e)}")

    response_metadata = {
        "search_type": "Quantum (Grover's Algorithm)",
        "request_list_size_N": request.list_size_N,
        "request_target_number": request.target_number,
        "simulation_qubits_used": total_qubits,
        "simulation_search_space_N": N_sim_space,
        "optimal_iterations_calculated": optimal_iterations,
        "total_iterations_simulated": max_iterations_to_run
    }

    return {
        "metadata": response_metadata,
        "evolution_by_iteration": evolution_data
    }


# --- Endpoint 2: Căutare CLASICĂ (Liniară) ---
@app.post("/search/classical", response_model=ClassicalSearchResponse, tags=["Classical"])
async def search_with_classical(request: GroverSearchRequest):
    """
    Simulează o căutare clasică liniară (O(N)).

    Iterează prin listă element cu element (de la 0 la N-1)
    până găsește ținta și raportează numărul de pași.
    """
    validate_request(request)

    search_log = []
    steps_taken = 0
    found = False

    # Simularea căutării clasice
    # Iterează prin lista conceptuală de la 0 la N-1
    for i in range(request.list_size_N):
        steps_taken += 1  # Un pas = o "verificare"
        current_number = i

        if current_number == request.target_number:
            found = True
            search_log.append(
                {"step": steps_taken, "checking_number": current_number, "found": True}
            )
            break  # Am găsit-o, ne oprim
        else:
            search_log.append(
                {"step": steps_taken, "checking_number": current_number, "found": False}
            )

    response_metadata = {
        "search_type": "Classical Linear Search (O(N))",
        "request_list_size_N": request.list_size_N,
        "request_target_number": request.target_number,
        "steps_taken_to_find": steps_taken,
        "found": found
    }

    return {
        "metadata": response_metadata,
        "search_log": search_log
    }


# =======================================================
#  RULAREA SERVERULUI
# =======================================================

if __name__ == "__main__":
    import uvicorn

    # Rulează serverul pe http://127.0.0.1:8000
    uvicorn.run(app, host="127.0.0.1", port=8000)