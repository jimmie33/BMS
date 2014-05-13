
SRCDIR=./src
INCLDIR=./src $(HOME)/local/include
LIBDIR=$(HOME)/local/lib
LIBS=opencv_core opencv_highgui opencv_imgproc
CFLAGS=-Wall -O2 -fopenmp $(patsubst %,-I%,$(INCLDIR))
LDFLAGS=$(patsubst %,-L%,$(LIBDIR)) $(patsubst %,-l%,$(LIBS))
SOURCES=main.cpp BMS.cpp
OBJECTS=$(SOURCES:.cpp=.o)
EXECUTABLE=BMS

all: $(EXECUTABLE)

$(EXECUTABLE): $(OBJECTS)
	g++ $(CFLAGS) $(LDFLAGS) -o $@ $^

%.o: $(SRCDIR)/%.cpp
	g++ -c $(CFLAGS) $< -o $@


clean: 
	rm -rf *o $(EXECUTABLE)