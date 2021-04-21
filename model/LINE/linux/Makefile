run:
	g++ line.cpp -o line-test -lgsl -lgslcblas
	./line-test -train ./net.txt -output embedding_out.txt -binary 1 -size 200 -order 2 -negative 5 -samples 100 -rho 0.025 -threads 8
norm:
	rm normalize
	g++ normalize.cpp -o normalize    
	./normalize -input embedding_out.txt -output embedding_out_norm.txt -binary 0