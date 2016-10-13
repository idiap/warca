CXX = g++
CFLAGS = -Wall -fPIC -fopenmp
SRCDIR = src
INCLUDES = src
all: warca_train_double_precision warca_train_single_precision warca_predict_double_precision warca_predict_single_precision 

warca_train_double_precision:  
	$(CXX) $(CFLAGS) -I$(INCLUDES)  ${SRCDIR}/math_functions.cc ${SRCDIR}/random.cc ${SRCDIR}/warca.cc ${SRCDIR}/warca_train_double_precision.cc -o warca_train_double_precision.bin -lm -lcblas -std=c++11
warca_train_single_precision:  
	$(CXX) $(CFLAGS) -I$(INCLUDES)  ${SRCDIR}/math_functions.cc ${SRCDIR}/random.cc ${SRCDIR}/warca.cc ${SRCDIR}/warca_train_single_precision.cc -o warca_train_single_precision.bin -lm -lcblas -std=c++11
warca_predict_double_precision:  
	$(CXX) $(CFLAGS) -I$(INCLUDES)  ${SRCDIR}/math_functions.cc ${SRCDIR}/random.cc ${SRCDIR}/warca.cc ${SRCDIR}/warca_predict_double_precision.cc -o warca_predict_double_precision.bin -lm -lcblas -std=c++11
warca_predict_single_precision:  
	$(CXX) $(CFLAGS) -I$(INCLUDES)  ${SRCDIR}/math_functions.cc ${SRCDIR}/random.cc ${SRCDIR}/warca.cc ${SRCDIR}/warca_predict_single_precision.cc -o warca_predict_single_precision.bin -lm -lcblas -std=c++11
clean:
	rm -f warca.o warca_train_double_precision.bin warca_train_single_precision.bin warca_predict_double_precision.bin warca_predict_single_precision.bin 
