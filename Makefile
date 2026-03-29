.PHONY: all clean goby notebook

all: goby

goby:
	$(MAKE) -C csrc

clean:
	$(MAKE) -C csrc clean

notebook:
	python3 tools/make_colab.py
