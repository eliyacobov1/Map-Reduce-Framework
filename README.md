# Map-Reduce-Framework
An implementation of Map-Reduce framework in c++.

This framework recieves a client as defined in the file Client.h, 
with the predefined 
functions 'map' and 'reduce'. It then proccess the given data 
and places it in an output vector. 

* (K1, V1), (K2, v2), (k3, v3) are the key-value expected types by the client for
  each of the phases (mapping, sorting and reducing)

* Multithreading is maintained throughout the whole process. 
###   To initialize the process, call the function 'initializeJob' 
###   in the file main.cpp with a valid Client that follows the pattern 
###   of Client.h



