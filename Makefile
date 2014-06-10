main: main.cpp AbstractLayer.o Layer.o ConvolutionLayer.o SamplingLayer.o NN.o
	g++ -Wall -o main main.cpp AbstractLayer.o Layer.o ConvolutionLayer.o SamplingLayer.o NN.o
NN.o: NN.h NN.cpp
	g++ -c NN.cpp
ConvolutionLayer.o: ConvolutionLayer.h ConvolutionLayer.cpp
	g++ -c ConvolutionLayer.cpp
SamplingLayer.o: SamplingLayer.h SamplingLayer.cpp
	g++ -c SamplingLayer.cpp
Layer.o: Layer.h Layer.cpp
	g++ -c Layer.cpp
AbstractLayer.o: Share.h AbstractLayer.h AbstractLayer.cpp
	g++ -c AbstractLayer.cpp
clean:
	rm *.o main
# height width labelcount samplecount testcount
run: main
	./main 32 32 10 20000 40000 train.csv test.csv
test: main
	./main 32 32 10 1000 40000 train.csv test.csv
