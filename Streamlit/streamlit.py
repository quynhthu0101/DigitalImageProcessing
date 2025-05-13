import cv2
import numpy as np
import streamlit as st
from PIL import Image
from ultralytics import YOLO
from ultralytics.utils.plotting import Annotator

import chapter3 as c3
import chapter4 as c4
import chapter9 as c9
import os
import onnxruntime as ort


class ImageProcessingApp:
    def __init__(self):
        self.st = st
        self.chapter = ""
        self.function = ""
        self.uploaded_file = None
        self.img_cv = None
        self.imgout = None

    def web_ui(self):
        """Set up the web UI for the app."""
        self.st.set_page_config(
            page_title="Ứng dụng xử lý ảnh số", layout="wide"
        )
        # Display snow only if it's the first time the app is loaded
        if 'snow_displayed' not in self.st.session_state:
            self.st.snow()
            self.st.session_state.snow_displayed = True

        # Custom dark style and animations
        self.st.markdown("""
            <style>
                html, body, [class*="css"] {
                    background-color: #111 !important;
                    color: #eee !important;
                    font-family: 'Segoe UI', sans-serif;
                    transition: all 0.3s ease;
                }
                .stApp {
                    background-color: #111;
                    padding: 20px;
                    padding-bottom: 60px; /* Giữ khoảng trống cho footer */
                }
                /* Title styling */
                .app-title {
                    text-align: center;
                    font-size: 2.5rem;
                    font-weight: bold;
                    color: #FF64DA; /* Màu hồng */
                    margin-bottom: 30px;
                    animation: fadeIn 1.2s ease-in-out;
                }
                /* Sidebar */
                section[data-testid="stSidebar"] {
                    background-color: #1e1e1e;
                    border-right: 1px solid #333;
                }
                /* Image fade-in + scale */
                .stImage > img {
                    border-radius: 10px;
                    box-shadow: 0 5px 20px rgba(0, 255, 204, 0.2);
                    animation: scaleIn 0.7s ease;
                    transition: transform 0.3s;
                }
                .stImage:hover > img {
                    transform: scale(1.02);
                }
                @keyframes fadeIn {
                    0% {opacity: 0; transform: translateY(-20px);}
                    100% {opacity: 1; transform: translateY(0);}
                }
                @keyframes scaleIn {
                    0% {opacity: 0; transform: scale(0.9);}
                    100% {opacity: 1; transform: scale(1);}
                }

                /* Footer */
                #custom-footer {
                    position: fixed;
                    bottom: 0;
                    width: 100%;
                    background-color: #111;
                    color: #eee;
                    text-align: center;
                    padding: 10px 0;
                    font-size: 0.9rem;
                }
            </style>
        """, unsafe_allow_html=True)

        # Tiêu đề chính màu hồng
        self.st.markdown(
            '<div class="app-title">Ứng dụng xử lý ảnh số</div>', unsafe_allow_html=True
        )

        self.st.sidebar.markdown(f"""
            <div style="padding-left: 1cm; margin-bottom: -80px; margin-top:-2cm;">
                <img src="data:image/png;base64,{self.get_base64_of_image('image/logo.png')}" 
                    style="width: 200px; border-radius: 3px;" />
            </div>
        """, unsafe_allow_html=True)

        # Footer
        self.st.markdown(
            """
            <footer id="custom-footer">
                Made with ❤️ | Image Processing App | 2025
            </footer>
            """,
            unsafe_allow_html=True
        )

    def get_base64_of_image(self, path):
        import base64
        with open(path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode()

    def sidebar(self):
        with self.st.sidebar:
            self.st.header("Chọn chương và chức năng")
            # Chọn chương
            self.chapter = self.st.selectbox("📚 Chọn chương", [
                "",
                "Nhận diện khuôn mặt",
                "Nhận diện trái cây",
                "Chapter 3",
                "Chapter 4",
                "Chapter 9",
                "Phân biệt trái cây tươi hay hỏng",
                "Nhận diện biển báo giao thông"
            ])

            # Nếu là nhận diện khuôn mặt,... không hiển thị gì nữa
            if self.chapter in ["Nhận diện khuôn mặt", "Phân biệt trái cây tươi hay hỏng", "Nhận diện biển báo giao thông"]:
                return  # Không hiển thị thêm bất kỳ lựa chọn nào

            # Nếu người dùng đã chọn chương (ngoài chương đặc biệt), hiển thị chế độ xem kết quả
            if self.chapter:
                # Nếu chọn YOLO, tự động chọn chế độ "Ảnh gốc và kết quả"
                if self.chapter == "Nhận diện trái cây":
                    self.view_mode = "Ảnh gốc và kết quả"
                else:
                    # Chọn chế độ xem kết quả
                    self.view_mode = self.st.selectbox("⚙️ Chọn chế độ xem kết quả", [
                                                       "Ảnh gốc và kết quả", "Comparison"])

                # Chọn chức năng theo chương đã chọn
                function_list = []
                if self.chapter == "Chapter 3":
                    function_list = ["Negative", "NegativeColor", "Logarit", "Power", "PiecewiseLine",
                                     "Histogram", "HistEqual", "HistEqualColor", "LocalHist", "HistStat",
                                     "SmoothBox", "SmoothGauss", "MedianFilter", "Sharpening", "SharpeningMask", "Gradient"]
                elif self.chapter == "Chapter 4":
                    function_list = ["Spectrum", "RemoveMorieSimple", "RemoveInterference",
                                     "CreateMotion", "Demotion", "DemotionNoise"]
                elif self.chapter == "Chapter 9":
                    function_list = ["Erosion",
                                     "Dilation", "Boundary", "Contour"]
                self.function = self.st.selectbox(
                    "⚙️ Chọn chức năng", [""] + function_list if function_list else [])

    def upload_image(self):
        if self.chapter:
            self.uploaded_file = self.st.file_uploader(
                "🖼️ Tải ảnh lên", type=["jpg", "jpeg", "png", "bmp", "tif", "webp"], key="image_uploader"
            )

        if not self.uploaded_file:
            image_extensions = ["jpg", "jpeg", "png", "bmp", "tif", "webp"]
            found = False

            if self.function:
                for ext in image_extensions:
                    test_image_path = f"image/test/{self.function}.{ext}"
                    if os.path.exists(test_image_path):
                        self.uploaded_file = open(test_image_path, "rb")
                        found = True
                        break

            if not found:
                for ext in image_extensions:
                    test_image_path = f"image/test/{self.chapter}.{ext}"
                    if os.path.exists(test_image_path):
                        self.uploaded_file = open(test_image_path, "rb")
                        break

        if self.uploaded_file:
            img = Image.open(self.uploaded_file)
            img_array = np.array(img)
            if len(img_array.shape) == 2:
                if img_array.dtype == np.bool_:
                    img_array = img_array.astype(np.uint8) * 255
                self.img_cv = img_array
            else:
                self.img_cv = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)

    def process_image(self):
        if self.img_cv is None:
            return

        img = self.img_cv
        self.imgout = None

        try:
            # Danh sách các hàm yêu cầu ảnh màu và ảnh đen trắng
            grayscale_required = {
                "Negative", "Logarit", "Power", "PiecewiseLine", "Histogram",
                "HistEqual", "LocalHist", "HistStat",
                "SmoothBox", "SmoothGauss", "MedianFilter",
                "Sharpening", "SharpeningMask",
                "Gradient", "Demotion", "DemotionNoise",
                "Spectrum", "RemoveMorie", "RemoveInterference", "CreateMotion", "Demotion",
                "Erosion", "Dilation", "Boundary", "Contour"
            }

            color_required = {
                "NegativeColor", "HistEqualColor"
            }

            # Xác định ảnh đầu vào có phải grayscale không
            is_grayscale = len(img.shape) == 2 or (
                len(img.shape) == 3 and img.shape[2] == 1)

            # Xác định ảnh đầu vào có phải ảnh màu
            is_color = len(img.shape) == 3 and img.shape[2] == 3

            # Kiểm tra điều kiện ảnh màu/xám trước khi xử lý
            if self.function in grayscale_required and not is_grayscale:
                self.st.warning(
                    "⚠️ Chức năng này yêu cầu **ảnh đen trắng (grayscale)**. Vui lòng chọn lại ảnh.")
                return
            if self.function in color_required and not is_color:
                self.st.warning(
                    "⚠️ Chức năng này yêu cầu **ảnh màu (RGB)**. Vui lòng chọn lại ảnh.")
                return

            # Xử lý theo chương
            if self.chapter == "Chapter 3":
                func_map = {
                    "Negative": c3.Negative, "NegativeColor": c3.NegativeColor,
                    "Logarit": c3.Logarit, "Power": c3.Power,
                    "PiecewiseLine": c3.PiecewiseLine, "Histogram": c3.Histogram,
                    "HistEqual": lambda x: cv2.equalizeHist(x),
                    "HistEqualColor": c3.HistEqualColor, "LocalHist": c3.LocalHist,
                    "HistStat": c3.HistStat,
                    "SmoothBox": lambda x: cv2.boxFilter(x, cv2.CV_8UC1, (21, 21)),
                    "SmoothGauss": lambda x: cv2.GaussianBlur(x, (43, 43), 7.0),
                    "MedianFilter": lambda x: cv2.medianBlur(x, 5),
                    "Sharpening": c3.Sharpening, "SharpeningMask": c3.SharpeningMask,
                    "Gradient": c3. Gradient
                }
                if self.function in func_map:
                    self.imgout = func_map[self.function](img)

            elif self.chapter == "Chapter 4":
                if self.function == "DemotionNoise":
                    temp = cv2.medianBlur(img, 7)
                    self.imgout = c4.Demotion(temp)
                else:
                    func_map = {
                        "Spectrum": c4.Spectrum, "RemoveMorieSimple": c4.RemoveMorie,
                        "RemoveInterference": c4.RemoveInterference, "CreateMotion": c4.CreateMotion,
                        "Demotion": c4.Demotion
                    }
                    if self.function in func_map:
                        self.imgout = func_map[self.function](img)

            elif self.chapter == "Chapter 9":
                func_map = {
                    "Erosion": c9.Erosion, "Dilation": c9.Dilation,
                    "Boundary": c9.Boundary, "Contour": c9.Contour
                }
                if self.function in func_map:
                    self.imgout = func_map[self.function](img)

            elif self.chapter == "Nhận diện trái cây" or self.chapter:
                # Kiểm tra nếu ảnh không phải ảnh màu
                if not is_color:
                    self.st.warning(
                        "⚠️Yêu cầu **ảnh màu (RGB)** để phát hiện đối tượng. Vui lòng chọn ảnh khác.")
                    return

                self.imgout = img.copy()
                model = YOLO("model/best.onnx")
                results = model.predict(img, conf=0.5, verbose=False)
                annotator = Annotator(self.imgout)
                names = model.names
                boxes = results[0].boxes.xyxy.cpu()
                clss = results[0].boxes.cls.cpu().tolist()
                confs = results[0].boxes.conf.tolist()
                for box, cls, conf in zip(boxes, clss, confs):
                    label = f"{names[int(cls)]} {conf:.2f}"
                    annotator.box_label(box, label, color=(
                        255, 255, 255), txt_color=(255, 0, 0))
                self.imgout = annotator.result()
        except Exception as e:
            self.st.error(f"❌ Đã xảy ra lỗi khi xử lý ảnh: {e}")

    def display_output(self):
        if self.img_cv is not None and self.imgout is not None:
            # Nếu chọn chế độ Ảnh gốc và kết quả
            if self.view_mode == "Ảnh gốc và kết quả":
                col1, col2 = self.st.columns(2)
                with col1:
                    self.st.image(
                        cv2.cvtColor(self.img_cv, cv2.COLOR_BGR2RGB) if len(
                            self.img_cv.shape) == 3 else self.img_cv,
                        caption="🖼️ Ảnh gốc", use_container_width=True
                    )
                with col2:
                    self.st.image(
                        cv2.cvtColor(self.imgout, cv2.COLOR_BGR2RGB) if len(
                            self.imgout.shape) == 3 else self.imgout,
                        caption="📸 Kết quả xử lý", use_container_width=True
                    )
            # Nếu chọn chế độ Comparison
            elif self.view_mode == "Comparison" and self.chapter != "Nhận diện trái cây":
                from streamlit_image_comparison import image_comparison

                # Chuyển ảnh từ BGR sang RGB trước khi hiển thị trong chế độ comparison
                img1_rgb = cv2.cvtColor(self.img_cv, cv2.COLOR_BGR2RGB) if len(
                    self.img_cv.shape) == 3 else self.img_cv
                img2_rgb = cv2.cvtColor(self.imgout, cv2.COLOR_BGR2RGB) if len(
                    self.imgout.shape) == 3 else self.imgout

                # Hiển thị ảnh so sánh
                static_component = image_comparison(
                    img1=img1_rgb,
                    img2=img2_rgb,
                    label1="Ảnh gốc",
                    label2="Kết quả xử lý",
                    width=700
                )

    def predict_fruit_freshness(self):
        session = ort.InferenceSession("model/model.onnx")

        if self.img_cv is None:
            return

        # Tiền xử lý ảnh: resize thành (150, 150), chuẩn hóa và giữ nguyên định dạng NHWC
        img = self.img_cv  # <-- Thay img bằng self.img_cv trực tiếp
        img_resized = cv2.resize(img, (150, 150))
        img_normalized = img_resized.astype(np.float32) / 255.0  # [0,1]

        # Thêm batch dimension -> shape: [1, 150, 150, 3]
        img_input = np.expand_dims(img_normalized, axis=0)

        # Kiểm tra kiểu dữ liệu đầu vào của mô hình
        input_type = session.get_inputs()[0].type
        if input_type == "tensor(float)":
            img_input = img_input.astype(np.float32)
        else:
            raise ValueError(f"Unsupported input type: {input_type}")

        # Dự đoán
        input_name = session.get_inputs()[0].name
        outputs = session.run(None, {input_name: img_input})

        prediction = float(outputs[0][0][0])  # Kết quả là xác suất hỏng

        # Hiển thị kết quả
        col1, col2 = self.st.columns(2)
        with col1:
            self.st.image(cv2.cvtColor(self.img_cv, cv2.COLOR_BGR2RGB),
                          caption="Ảnh gốc", use_container_width=True)

        with col2:
            if prediction > 0.5:
                self.st.error(
                    f"❌ Kết quả: Trái cây **hỏng** ({prediction*100:.2f}%)")
            else:
                self.st.success(
                    f"✅ Kết quả: Trái cây **tươi** ({(1 - prediction)*100:.2f}%)")

    def run_face_detection_video(self):
        self.st.info(
            "📹 Nhấn 'Start' để bật webcam và nhận diện khuôn mặt từng người.")
        start = self.st.button("▶️ Start")

        if start:
            try:
                # Load mô hình
                detector = cv2.FaceDetectorYN.create(
                    model="model/face_detection_yunet_2023mar.onnx",
                    config="",
                    input_size=(320, 320),
                    score_threshold=0.9,
                    nms_threshold=0.3,
                    top_k=5000,
                    backend_id=0,
                    target_id=0,
                )

                recognizer = cv2.FaceRecognizerSF.create(
                    "model/face_recognition_sface_2021dec.onnx", "")

                # Load mô hình SVM và danh sách tên người
                import joblib
                svc = joblib.load("model/svc.pkl")
                # Danh sách tên tương ứng label
                mydict = ['DiemQuynh', 'KaPhuc', 'LamHieu', 'QuynhThu', 'ThuHang']

                # Khởi tạo webcam
                cap = cv2.VideoCapture(0)
                frame_placeholder = self.st.empty()

                while cap.isOpened():
                    ret, frame = cap.read()
                    if not ret:
                        break

                    h, w, _ = frame.shape
                    detector.setInputSize((w, h))
                    _, faces = detector.detect(frame)

                    if faces is not None:
                        for face in faces:
                            coords = list(map(int, face[:4]))  # (x, y, w, h)
                            face_crop = frame[coords[1]:coords[1] +
                                              coords[3], coords[0]:coords[0]+coords[2]]

                            if face_crop.size == 0:
                                continue

                            # Align face
                            aligned_face = recognizer.alignCrop(frame, face)
                            # Extract features
                            face_feature = recognizer.feature(aligned_face)
                            # Predict with SVM
                            pred = svc.predict(face_feature)
                            name = mydict[int(pred[0])] if int(
                                pred[0]) < len(mydict) else "Unknown"

                            # Vẽ box và tên
                            cv2.rectangle(
                                frame, (coords[0], coords[1]), (coords[0]+coords[2], coords[1]+coords[3]), (0, 255, 0), 2)
                            cv2.putText(
                                frame, name, (coords[0], coords[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    frame_placeholder.image(frame_rgb, channels="RGB")

                cap.release()

            except Exception as e:
                self.st.error(f"❌ Lỗi khi chạy webcam: {e}")

    def run_traffic_sign_detection(self):

        st.info("🚦 Nhấn vào Upload Ảnh hoặc Video để nhận diện biển báo giao thông.")

        model_path = "model/traffic_signs_classification.onnx"
        if not os.path.exists(model_path):
            st.error(f"❌ Không tìm thấy mô hình tại {model_path}")
            return

        ort_session = ort.InferenceSession(model_path)

        def preprocessing(img):
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            img = cv2.equalizeHist(img)
            img = img / 255.0
            return img

        def getClassName(classNo):
            classNames = [
                'Speed Limit 20 km/h', 'Speed Limit 30 km/h', 'Speed Limit 50 km/h',
                'Speed Limit 60 km/h', 'Speed Limit 70 km/h', 'Speed Limit 80 km/h',
                'End of Speed Limit 80 km/h', 'Speed Limit 100 km/h', 'Speed Limit 120 km/h',
                'No passing', 'No passing for vehicles over 3.5 metric tons',
                'Right-of-way at the next intersection', 'Priority road', 'Yield',
                'Stop', 'No vehicles', 'Vehicles over 3.5 metric tons prohibited',
                'No entry', 'General caution', 'Dangerous curve to the left',
                'Dangerous curve to the right', 'Double curve', 'Bumpy road',
                'Slippery road', 'Road narrows on the right', 'Road work',
                'Traffic signals', 'Pedestrians', 'Children crossing',
                'Bicycles crossing', 'Beware of ice/snow', 'Wild animals crossing',
                'End of all speed and passing limits', 'Turn right ahead',
                'Turn left ahead', 'Ahead only', 'Go straight or right',
                'Go straight or left', 'Keep right', 'Keep left',
                'Roundabout mandatory', 'End of no passing',
                'End of no passing by vehicles over 3.5 metric tons'
            ]
            return classNames[classNo]

        def process_image(img):
            img_processed = cv2.resize(img, (32, 32))
            img_processed = preprocessing(img_processed)
            img_processed = img_processed.reshape(1, 32, 32, 1).astype(np.float32)

            inputs = {ort_session.get_inputs()[0].name: img_processed}
            predictions = ort_session.run(None, inputs)

            classIndex = np.argmax(predictions[0])
            probabilityValue = np.amax(predictions[0])

            text_class = f"CLASS: {classIndex} {getClassName(classIndex)}"
            text_prob = f"PROBABILITY: {round(probabilityValue * 100, 2)}%"

            # Resize ảnh gốc về kích thước lớn hơn để hiển thị rõ ràng
            display_img = cv2.resize(img, (640, 480))
            cv2.putText(display_img, text_class, (20, 35), cv2.FONT_HERSHEY_SIMPLEX,
                        0.75, (0, 255, 0), 2, cv2.LINE_AA)
            cv2.putText(display_img, text_prob, (20, 75), cv2.FONT_HERSHEY_SIMPLEX,
                        0.75, (0, 255, 0), 2, cv2.LINE_AA)
            return display_img

        option = st.selectbox("Chọn nguồn hình ảnh", ["Upload Ảnh", "Upload Video"])

        if option == "Upload Ảnh":
            uploaded_file = st.file_uploader("Chọn ảnh", type=["jpg", "jpeg", "png"])
            if uploaded_file is not None:
                file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
                img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
                processed_img = process_image(img)
                st.image(processed_img, channels="BGR", caption="Kết quả nhận diện")

        elif option == "Upload Video":
            uploaded_video = st.file_uploader("Chọn video", type=["mp4", "avi", "mov", "gif"])
            if uploaded_video is not None:
                # Lưu file video tạm thời
                with open("temp_video.mp4", "wb") as f:
                    f.write(uploaded_video.read())
                
                # Mở video bằng OpenCV
                cap = cv2.VideoCapture("temp_video.mp4")
                
                # Kiểm tra xem video có mở được không
                if not cap.isOpened():
                    st.error("❌ Không thể mở video. Kiểm tra định dạng hoặc file video.")
                    return
                
                # Hiển thị video frame-by-frame
                stop_video = st.button("🛑 Dừng Video")
                frame_placeholder = st.empty()
                
                while not stop_video:
                    ret, frame = cap.read()
                    
                    # Nếu video kết thúc, đặt lại về khung hình đầu tiên
                    if not ret:
                        # Đặt lại video về khung hình đầu tiên
                        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                        continue
                    
                    # Xử lý khung hình
                    processed_frame = process_image(frame)
                    
                    # Hiển thị khung hình đã xử lý
                    frame_placeholder.image(processed_frame, channels="BGR", caption="Video đang xử lý")
                
                # Giải phóng tài nguyên
                cap.release()
                os.remove("temp_video.mp4")  # Xóa file video tạm thời
                st.success("✅ Đã dừng video.")

    def run(self):
        """Run the full app."""
        self.web_ui()
        self.sidebar()

        if self.chapter == "Nhận diện khuôn mặt":
            self.run_face_detection_video()
        elif self.chapter == "Nhận diện trái cây":
            self.upload_image()
            self.process_image()
            self.display_output()
        elif self.chapter == "Phân biệt trái cây tươi hay hỏng":
            self.upload_image()
            self.predict_fruit_freshness()
        elif self.chapter == "Nhận diện biển báo giao thông":  
            self.run_traffic_sign_detection()
        else:
            if self.chapter and self.function:
                self.upload_image()
                self.process_image()
                self.display_output()


if __name__ == "__main__":
    app = ImageProcessingApp()
    app.run()
