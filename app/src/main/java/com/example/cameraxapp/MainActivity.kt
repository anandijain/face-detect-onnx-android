package com.example.cameraxapp

import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import android.content.Context
import android.content.pm.PackageManager
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Paint
import android.os.Build
import android.os.Bundle
import android.util.AttributeSet
import android.view.View
import android.widget.Toast
import androidx.activity.ComponentActivity
import androidx.activity.result.contract.ActivityResultContracts
import androidx.camera.core.AspectRatio
import androidx.camera.core.CameraSelector
import androidx.camera.core.ImageAnalysis
import androidx.camera.core.ImageCapture
import androidx.camera.core.ImageProxy
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.core.content.ContextCompat
import com.example.cameraxapp.databinding.ActivityMainBinding
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.Job
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import java.util.concurrent.ExecutorService
import java.util.concurrent.Executors
import android.graphics.Bitmap
import android.util.Log
import java.nio.FloatBuffer
import java.util.Collections
import retrofit2.Retrofit
import retrofit2.converter.gson.GsonConverterFactory
import retrofit2.http.GET
import retrofit2.http.Query
import okhttp3.ResponseBody

const val DIM_BATCH_SIZE = 1
const val DIM_PIXEL_SIZE = 3
const val IMAGE_SIZE_X = 320
const val IMAGE_SIZE_Y = 240


interface ApiService {
    @GET("servo")
    suspend fun setServoAngles(
        @Query("angle1") angle1: Int,
        @Query("angle2") angle2: Int
    ): ResponseBody
}

object RetrofitClient {
    private val retrofit = Retrofit.Builder()
        .baseUrl("http://192.168.4.31/")  // Replace with the IP address of your ESP32
        .addConverterFactory(GsonConverterFactory.create())
        .build()

    val apiService: ApiService = retrofit.create(ApiService::class.java)
}

class MainActivity : ComponentActivity() {
    private lateinit var viewBinding: ActivityMainBinding
    private lateinit var overlayView: OverlayView

    private val backgroundExecutor: ExecutorService by lazy { Executors.newSingleThreadExecutor() }
    private val scope = CoroutineScope(Job() + Dispatchers.Main)

    private var ortEnv: OrtEnvironment? = null
    private var ortSess: OrtSession? = null
    private var imageCapture: ImageCapture? = null
    private var imageAnalysis: ImageAnalysis? = null

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        viewBinding = ActivityMainBinding.inflate(layoutInflater)
        setContentView(viewBinding.root)
        overlayView = findViewById(R.id.overlayView)
        ortEnv = OrtEnvironment.getEnvironment()

        if (!hasPermissions(baseContext)) {
            activityResultLauncher.launch(REQUIRED_PERMISSIONS)
        } else {
            startCamera()
        }
    }

    private fun startCamera() {
        val cameraProviderFuture = ProcessCameraProvider.getInstance(this)
        val cameraProvider: ProcessCameraProvider = cameraProviderFuture.get()
        val cameraSelector = CameraSelector.DEFAULT_FRONT_CAMERA

        val preview = androidx.camera.core.Preview.Builder()
            .setTargetAspectRatio(AspectRatio.RATIO_16_9)
            .build()
            .also {
                it.setSurfaceProvider(viewBinding.previewView.surfaceProvider)
            }

        imageCapture = ImageCapture.Builder()
            .setTargetAspectRatio(AspectRatio.RATIO_16_9)
            .build()
        imageAnalysis = ImageAnalysis.Builder()
            .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
            .build()

        cameraProvider.unbindAll()

        cameraProvider.bindToLifecycle(
            this, cameraSelector, preview, imageCapture, imageAnalysis
        )

        setORTAnalyzer()
    }

    private val activityResultLauncher =
        registerForActivityResult(ActivityResultContracts.RequestMultiplePermissions())
        { permissions ->
            var permissionGranted = true
            permissions.entries.forEach {
                if (it.key in REQUIRED_PERMISSIONS && !it.value) {
                    permissionGranted = false
                }
            }
            if (!permissionGranted) {
                Toast.makeText(
                    baseContext,
                    "Permission request denied",
                    Toast.LENGTH_SHORT
                ).show()
            } else {
                startCamera()
            }
        }

    companion object {
        private const val TAG = "CameraXApp"
        private const val FILENAME_FORMAT = "yyyy-MM-dd-HH-mm-ss-SSS"
        private val REQUIRED_PERMISSIONS =
            mutableListOf(
                android.Manifest.permission.CAMERA
            ).apply {
                if (Build.VERSION.SDK_INT <= Build.VERSION_CODES.P) {
                    add(android.Manifest.permission.WRITE_EXTERNAL_STORAGE)
                }

            }.toTypedArray()

        fun hasPermissions(context: Context) = REQUIRED_PERMISSIONS.all {
            ContextCompat.checkSelfPermission(context, it) == PackageManager.PERMISSION_GRANTED
        }
    }

    private fun updateUI(result: List<Float>) {
        if (result.isEmpty()) return
        val left = (1 - result[2]) * overlayView.width
        val top = result[1] * overlayView.height
        val right = (1 - result[0]) * overlayView.width
        val bottom = result[3] * overlayView.height

        // Calculate the center of the bounding box
        val centerX = (left + right) / 2
        val centerY = (top + bottom) / 2

        // Map the center coordinates to angles (0 to 180)
        val angleX = (centerX / overlayView.width) * 180
        val angleY = (centerY / overlayView.height) * 180

        // Convert to integer values
        val angle1Int = angleX.toInt()
        val angle2Int = angleY.toInt()

        // Make the GET request using Retrofit
        CoroutineScope(Dispatchers.IO).launch {
            try {
                val response = RetrofitClient.apiService.setServoAngles(angle1Int, angle2Int).string()
                withContext(Dispatchers.Main) {
                    // Handle the response
//                    responseText = response
                    Log.d("MainActivity", "Response: $response")
                }
            } catch (e: Exception) {
                withContext(Dispatchers.Main) {
                    // Handle the error
//                    responseText = e.message ?: "Error"
                    Log.d("MainActivity", "Response: $e.message")

                }
            }
        }

        runOnUiThread {
            overlayView.updateRect(left, top, right, bottom)
        }
    }


    private suspend fun readModel(): ByteArray = withContext(Dispatchers.IO) {
        val modelID = R.raw.face_detector
        resources.openRawResource(modelID).readBytes()
    }

    private suspend fun createOrtSession(): OrtSession? = withContext(Dispatchers.Default) {
        ortSess = ortEnv?.createSession(readModel())
        ortSess
    }

    private fun setORTAnalyzer() {
        scope.launch {
            imageAnalysis?.clearAnalyzer()
            imageAnalysis?.setAnalyzer(
                backgroundExecutor,
                ORTAnalyzer(createOrtSession(), ::updateUI)
            )
        }
    }

    internal class ORTAnalyzer(
        private val ortSession: OrtSession?,
        private val callBack: (List<Float>) -> Unit
    ) : ImageAnalysis.Analyzer {
        override fun analyze(image: ImageProxy) {
            val imgBitmap = image.toBitmap()

            val inputName = ortSession?.inputNames?.iterator()?.next()
            val env = OrtEnvironment.getEnvironment()
            env.use {
                val tensor = createTensorFromImage(imgBitmap)
                tensor.use {
                    val results = ortSession?.run(Collections.singletonMap(inputName, tensor))
                    results.use {
                        val scores =
                            (results?.get(0)?.value as Array<Array<FloatArray>>)[0]
                        val boxes =
                            ((results.get(1)?.value) as Array<Array<FloatArray>>)[0]
                        val filtered_boxes = nonMaxSuppression(boxes, scores, 0.5f, 0.5f)

                        if (filtered_boxes.isEmpty()) {
                            image.close()
                            return
                        }
                        val result = filtered_boxes.first()
                        callBack(result)
                    }
                }
            }
            image.close()
        }

        protected fun finalize() {
            ortSession?.close()
        }
    }
}

class OverlayView(context: Context, attrs: AttributeSet?) : View(context, attrs) {
    private val paint = Paint().apply {
        color = Color.RED
        style = Paint.Style.STROKE
        strokeWidth = 5f
    }

    private var left = 0f
    private var top = 0f
    private var right = 0f
    private var bottom = 0f

    override fun onDraw(canvas: Canvas) {
        super.onDraw(canvas)
        canvas.drawRect(left, top, right, bottom, paint)
    }

    fun updateRect(left: Float, top: Float, right: Float, bottom: Float) {
        this.left = left
        this.top = top
        this.right = right
        this.bottom = bottom
        invalidate() // Request a redraw
    }
}

private fun createTensorFromImage(imageBitmap: Bitmap): OnnxTensor {
    val width = 320
    val height = 240
    val input = FloatArray(1 * 3 * height * width)
    val bitmap = imageBitmap.let { Bitmap.createScaledBitmap(imageBitmap, 320, 240, false) }

    val pixels = IntArray(width * height)
    bitmap.getPixels(pixels, 0, width, 0, 0, width, height)

    for (i in 0 until height) {
        for (j in 0 until width) {
            val idx = i * width + j
            val pixel = pixels[idx]
            val r = (pixel shr 16 and 0xFF) / 255.0f
            val g = (pixel shr 8 and 0xFF) / 255.0f
            val b = (pixel and 0xFF) / 255.0f
            input[(0 * height + i) * width + j] = r
            input[(1 * height + i) * width + j] = g
            input[(2 * height + i) * width + j] = b
        }
    }

    val env = OrtEnvironment.getEnvironment()
    val shape = longArrayOf(1, 3, height.toLong(), width.toLong())
    return OnnxTensor.createTensor(env, FloatBuffer.wrap(input), shape)
}

fun nonMaxSuppression(
    boxes: Array<FloatArray>,
    scores: Array<FloatArray>,
    scoreThreshold: Float,
    iouThreshold: Float
): List<List<Float>> {
    val filteredBoxes = mutableListOf<List<Float>>()

    // Filter out boxes with low scores
    val candidates = mutableListOf<List<Float>>()
    for (i in scores.indices) {
        if (scores[i][1] > scoreThreshold) {
            candidates.add(
                listOf(
                    boxes[i][0],
                    boxes[i][1],
                    boxes[i][2],
                    boxes[i][3],
                    scores[i][1]
                )
            )
        }
    }

    // Sort candidates by score in descending order
    candidates.sortByDescending { it[4] }

    // Perform Non-Maximum Suppression
    while (candidates.isNotEmpty()) {
        val (x1, y1, x2, y2, score) = candidates.removeAt(0)
        filteredBoxes.add(listOf(x1, y1, x2, y2, score))

        candidates.removeAll { box ->
            val iou = intersectionOverUnion(
                listOf(x1, y1, x2, y2),
                listOf(box[0], box[1], box[2], box[3])
            )
            iou >= iouThreshold
        }
    }

    return filteredBoxes
}

fun intersectionOverUnion(boxA: List<Float>, boxB: List<Float>): Float {
    val xA = maxOf(boxA[0], boxB[0])
    val yA = maxOf(boxA[1], boxB[1])
    val xB = minOf(boxA[2], boxB[2])
    val yB = minOf(boxA[3], boxB[3])

    val interArea = maxOf(0f, xB - xA) * maxOf(0f, yB - yA)
    val boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    val boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])

    return interArea / (boxAArea + boxBArea - interArea)
}
