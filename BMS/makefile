
SRCDIR=./src
INCLDIR=./src $(HOME)/local/include
LIBDIR=$(HOME)/local/lib
LIBS=libopencv_core libopencv_highgui libopencv_imgproc
CFLAGS=-Wall -O2 $(patsubst %,-I%,$(INCLDIR))
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