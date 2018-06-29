from wheel_class import wheel, controlWheels

def setup():
	w1 = wheel(4, 3, 2, 19)
	w2 = wheel(17, 27, 22, 19)
	w3 = wheel(13, 6, 5, 19)

	w1.initGPIO()
	w2.initGPIO()
	w3.initGPIO()

	controller = controlWheels(w1, w2, w3)
	return controller 

