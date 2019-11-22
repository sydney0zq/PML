#
# Makefile
# qiang.zhou, 2018-07-06 10:38
#

clean:
	@echo "Clean last experiments assets"
	#mv experiments/training legacy/training_`date +%Y-%m-%d_%H.%M`
	/bin/rm experiments/training -rf
	/bin/rm experiments/model/*.model -rf
	/bin/rm test_output -rf
