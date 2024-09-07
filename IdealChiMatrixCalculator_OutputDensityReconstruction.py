import numpy as np

# Use the operations variable to define the process undergone using Jones Calculus
# If the theoretical Chi matrix is known, uncomment chi_mat and insert the known values (Line 127)

#Defining possible operations for arbitrary angles

def qwp(theta_deg):
    """
    Returns the QWP (Quarter-Wave Plate) matrix for a given angle theta (in degrees).
    """

    theta = np.radians(theta_deg)
    # Define the complex number i
    i = 1j
    
    # Calculate cosine and sine of theta
    cos_theta = np.cos(theta)
    sin_theta = np.sin(theta)
    
    # Construct the QWP matrix using the provided formula
    qwp = np.exp(-i * np.pi / 4) * np.array([
        [cos_theta**2 + i*sin_theta**2, (1-i)*sin_theta*cos_theta],
        [(1-i)*sin_theta*cos_theta, sin_theta**2 + i*cos_theta**2]
    ])
    
    return qwp

def hwp(theta_deg):
    """
    Returns the HWP (Half-Wave Plate) matrix for a given angle theta (in degrees).
    """
    theta = np.radians(theta_deg)
    # Define the complex number i
    i = 1j
    
    # Calculate cosine and sine of theta
    cos_theta = np.cos(theta)
    sin_theta = np.sin(theta)
    
    # Construct the QWP matrix using the provided formula
    hwp = np.exp(-i * np.pi / 2) * np.array([
        [cos_theta**2 - sin_theta**2, 2*sin_theta*cos_theta],
        [2*sin_theta*cos_theta, sin_theta**2 - cos_theta**2]
    ])
    
    return hwp

def polarizer(theta_deg):
    """
    Returns the polarizer matrix for a given angle theta (in degrees).
    """

    # Convert theta from degrees to radians
    theta = np.radians(theta_deg)
    
    # Calculate cosine and sine of theta
    cos_theta = np.cos(theta)
    sin_theta = np.sin(theta)
    
    # Construct the polarizer matrix
    polarizer = np.array([
        [cos_theta**2, cos_theta * sin_theta],
        [cos_theta * sin_theta, sin_theta**2]
    ])
    
    return polarizer

#Identity
identity = np.array([
    [1, 0],
    [0, 1]
])

# We define our map as quantum_channel(rho, kraus_operator)
def quantum_channel(rho, kraus_operator):
    """
    Applies the quantum channel described by the Kraus operators to the input state.

    Parameters:
    - rho: The input density matrix (2x2 or larger numpy array).
    - kraus_operators: A list of Kraus operators (numpy arrays).

    Returns:
    - The output density matrix after applying the quantum process.
    """
    output_rho = np.zeros_like(rho, dtype=complex) 
    output_rho += kraus_operator @ rho @ kraus_operator.T.conj()
    return output_rho

#
# Insert the optics to be characterized. Matrix multiplication is @.
#

operations = polarizer(0) @ qwp(0)

#Following Example 8.5 in Nielsen & Chuang's Quantum Computation and Quantum Information 

p1 = np.array([
    [1, 0],
    [0, 0]
])
p1_prime = quantum_channel(p1, operations)
p4_prime = quantum_channel(np.array([[0 , 1], [1, 0]]) @ p1 @ np.array([[0 , 1], [1, 0]]), operations)
p2_prime= quantum_channel(p1 @ np.array([[0 , 1], [1, 0]]), operations)
p3_prime= quantum_channel(np.array([[0 , 1], [1, 0]]) @ p1, operations)

lam = (1/2) * np.array([
    [1, 0, 0, 1],
    [0, 1, 1, 0],
    [0, 1, -1, 0],
    [1, 0, 0, -1]
])

ps = np.block([
    [p1_prime, p2_prime],
    [p3_prime, p4_prime]
])

# Chi matrix of the ideal process
chi_mat = lam @ ps @ lam

# Eounding for better readability
chi_mat_rounded = np.round(chi_mat, 8)
print(f" Chi matrixR is :\n{chi_mat_rounded}")

# Insert ideal Chi matrix if known
# chi_mat = np.array([
#     [0.25, 0, 0, -0.25],
#     [0, 0, -0j, 0],
#     [0, 0j, 0, 0],
#     [-0.25, 0, 0, 0.25]
# ])

print("  ")
print("Result of running H, V, D, A, R and L through the process described by the Chi matrix")
print("  ")

# From the Chi matrix we reconstruct the output states to see if the Chi matrix makes sense

# Define the basis operators
I = np.eye(2)
sigma_x = np.array([[0, 1], [1, 0]])
sigma_y = np.array([[0, -1j], [1j, 0]])
sigma_z = np.array([[1, 0], [0, -1]])

basis_operators = [I, sigma_x, -1j * sigma_y, sigma_z] # following 8.5 in N&C

# Set numpy print options for better readability
np.set_printoptions(precision=3, suppress=True)

# Function to format and print the matrix nicely
def print_matrix(matrix, label, matrix_type):
    print(f"{matrix_type} density matrix for input state {label}:")
    for row in matrix:
        formatted_row = ["{: .3f}".format(elem) if np.isreal(elem) else "{: .3f}".format(elem) + "" for elem in row]
        print("[" + "  ".join(formatted_row) + "]")
    print()

eigenstates = [
    ("H", np.array([[1, 0], [0, 0]])),
    ("V", np.array([[0, 0], [0, 1]])),
    ("D", 0.5 * np.array([[1, 1], [1, 1]])),
    ("A", 0.5 * np.array([[1, -1], [-1, 1]])),
    ("R", 0.5 * np.array([[1, -1j], [1j, 1]])),
    ("L", 0.5 * np.array([[1, 1j], [-1j, 1]]))
]

# Calculate the output density matrix rho_out
rho_out = np.zeros((2, 2), dtype=complex)
for eigenstate in eigenstates:
    rho_out = np.zeros((2, 2), dtype=complex)
    rho = eigenstate[1]
    for m in range(4):
        for n in range(4):
            rho_out += chi_mat[m, n] * np.dot(np.dot(basis_operators[m], rho), basis_operators[n].conj().T) # changed so -i * pauli Y
    # print(f"Rho out for {eigenstate[0]} is {rho_out}")
    print_matrix(eigenstate[1], eigenstate[0], "Input")
    print_matrix(rho_out, eigenstate[0], "Output")

