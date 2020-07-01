import numpy as np
import cv2
from sklearn.mixture import GaussianMixture as GMM


def segmentar(img,n_components):

	# Ajusta a array da imagem para calcular a Mistura Gaussiana
	img_ar = img.reshape((-1,3))
	# Cria/treina o modelo com 2 segmentações baseado na array 'img_ar'
	model = GMM(n_components=n_components, covariance_type='tied').fit(img_ar)

	# Predição da Mistura Gaussiana
	segmented_pred = model.predict(img_ar)

	# Reshape para as dimensões da imagem original
	segmented = segmented_pred.reshape(img.shape[1],img.shape[0])

	result = segmented.astype(np.uint8)

	factor = 255 / float(result.max())

	final = result * factor
	final = final.astype(np.uint8)

	return final




img = cv2.imread("logo.png")

cv2.imshow("logo", img)
parte1 = segmentar(img,4)

cv2.imshow("parte1", parte1)

parte1_rgb = cv2.cvtColor(parte1,cv2.COLOR_GRAY2BGR)


parte2 = segmentar(parte1_rgb,3)

cv2.imshow("parte2", parte2)

invertida = np.invert(parte2)
cv2.imshow("invertida", invertida)

invertida_rgb = cv2.cvtColor(invertida,cv2.COLOR_GRAY2BGR)
parte3 = segmentar(invertida_rgb,2)

cv2.imshow("parte3", parte3)




img = cv2.cvtColor(parte3, cv2.COLOR_GRAY2BGR)
flag = cv2.imread("flag.jpg")
flag = cv2.resize(flag,(img.shape[0], img.shape[1]))
cv2.imshow("flag",flag)

mask = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

cv2.imshow("Imagem", img)

resultado1 = cv2.bitwise_and(img,flag, mask=mask)

cv2.imshow("resultado1", resultado1)

invert_mask = cv2.cvtColor(np.invert(mask), cv2.COLOR_GRAY2BGR)

final = cv2.addWeighted(invert_mask,1,resultado1,1,0)
cv2.imshow("final", final)
cv2.waitKey(0)

