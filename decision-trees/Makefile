JFLAGS = -cp lib/weka.jar:bin -d bin -sourcepath src
JAVAC = javac

.PHONY: directories
.SUFFIXES: .java .class

.java.class:
	$(JAVAC) $(JFLAGS) $*.java

CLASSES = src/FeatureGenerator.java src/Id3.java src/SGD.java src/DecisionStumps.java src/WekaTesterSGD.java src/WekaTesterTree.java src/WekaTesterDecisionStumps.java

all: directories classes

directories: 
	mkdir -p bin

classes: $(CLASSES:.java=.class)

clean:
	$(RM) -r bin

