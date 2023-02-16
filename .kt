class SimpleRNN {
    private val inputSize = 2
    private val outputSize = 1
    private val hiddenSize = 3

    private val w1 = randomMatrix(hiddenSize, inputSize)
    private val w2 = randomMatrix(outputSize, hiddenSize)

    private fun randomMatrix(rows: Int, cols: Int): Array<Array<Double>> {
        val matrix = Array(rows) { Array(cols) { 0.0 } }
        for (i in 0 until rows) {
            for (j in 0 until cols) {
                matrix[i][j] = Math.random()
            }
        }
        return matrix
    }

    fun forward(inputs: Array<Array<Double>>): Array<Array<Double>> {
        val hiddenLayer = Array(hiddenSize) { 0.0 }
        val outputLayer = Array(outputSize) { 0.0 }

        for (i in 0 until hiddenSize) {
            for (j in 0 until inputSize) {
                hiddenLayer[i] += w1[i][j] * inputs[0][j]
            }
        }

        for (i in 0 until outputSize) {
            for (j in 0 until hiddenSize) {
                outputLayer[i] += w2[i][j] * hiddenLayer[j]
            }
        }

        return outputLayer.map { arrayOf(it) }.toTypedArray()
    }
}
