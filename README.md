# Distributed Matrix Factorization using MPI

# Cómo ejecutar el código secuencial y paralelo

## Estructura del proyecto:


python3 secuencial.py --data_path u1m.data --N 10000
python3 secuencial.py --data_path u1m.data --N 100000
python3 secuencial.py --data_path u1m.data --N 200000
python3 secuencial.py --data_path u1m.data --N 500000

# 2. EJECUTAR VERSIÓN PARALELA CON MPI:

mpiexec -n 4 python paralelo.py --data_path u1m.data --N 10000 --num_threads 1 --sequential_time 2.729537

mpiexec -n 4 python paralelo.py --data_path u1m.data --N 100000 --num_threads 1 --sequential_time 25.067364

mpiexec -n 4 python paralelo.py --data_path u1m.data --N 200000 --num_threads 1 --sequential_time 50.321323

mpiexec -n 4 python paralelo.py --data_path u1m.data --N 500000 --num_threads 1 --sequential_time 122.438268

# misma N con distinto número de procesos:

mpiexec -n 1 python paralelo.py --data_path u1m.data --N 100000 --num_threads 1 --sequential_time 25.067364
mpiexec -n 2 python paralelo.py --data_path u1m.data --N 100000 --num_threads 1 --sequential_time 25.067364
mpiexec -n 4 python paralelo.py --data_path u1m.data --N 100000 --num_threads 1 --sequential_time 25.067364
mpiexec -n 8 python paralelo.py --data_path u1m.data --N 100000 --num_threads 1 --sequential_time 25.067364

# Si tu MPI da error de slots, usar:

mpiexec -n 1 python paralelo.py --data_path u1m.data --N 100000 --num_threads 1 --sequential_time 25.067364
