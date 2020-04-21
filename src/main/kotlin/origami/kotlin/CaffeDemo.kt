package origami.kotlin

import org.opencv.core.Core.minMaxLoc
import org.opencv.core.Scalar
import org.opencv.core.Size
import org.opencv.dnn.Dnn.*
import origami.Origami
import origami.Dnn
import org.opencv.dnn.Net

object CaffeeDemo {

    @Throws(Exception::class)
    @JvmStatic
    fun main(args: Array<String>) {
        Origami.init()
        val list = Dnn.readNetFromSpec("networks.caffe:convnet-age:1.0.0")
        val _net:Net = list.get(0) as Net
        val labels = list.get(2) as List<*>

        val image = Origami.urlToMat("https://i.pinimg.com/736x/cb/83/6c/cb836cc5f65b8c15ee1bfa94bd8e5c81.jpg")
        val inputBlob = blobFromImage(image, 1.0, Size(256.0, 256.0), Scalar(0.0), true, true)
        _net.setInput(inputBlob)
        _net.setPreferableBackend(DNN_BACKEND_OPENCV)
        val result = _net.forward()

        println(result.dump())

        val minmax = minMaxLoc(result)
        println(labels.get(minmax.maxLoc.x.toInt()))
    }

}
