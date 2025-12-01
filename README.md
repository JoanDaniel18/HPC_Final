# Distributed Matrix Factorization using MPI

# Cómo ejecutar el código secuencial y paralelo:

# 1. EJECUTAR VERSIÓN SECUENCIAL:
python3 secuencial.py --data_path u1m.data --N [10000, 100000, 200000, 500000]

# 2. EJECUTAR VERSIÓN PARALELA CON MPI:

Aquí X se reemplaza con el tiempo obtenido del script secuencial.

mpiexec -n 4 python paralelo.py --data_path u1m.data --N [10000,100000,200000, 500000] --num_threads 1 --sequential_time X

# misma N con distinto número de procesos:

mpiexec -n [1,2,4,8] python paralelo.py --data_path u1m.data --N 100000 --num_threads 1 --sequential_time X

# Si tu MPI da error de slots, usar:

mpiexec --oversubscribe -n 1 python paralelo.py --data_path u1m.data --N 100000 --num_threads 1 --sequential_time X
 
