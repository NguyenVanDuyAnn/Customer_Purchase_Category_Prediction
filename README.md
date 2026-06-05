# Customer Repurchase Prediction Dashboard

## Project Summary

## Project Summary

Trong lĩnh vực thương mại điện tử, việc dự đoán khả năng khách hàng quay lại mua hàng giúp doanh nghiệp tối ưu chiến lược marketing, cá nhân hóa trải nghiệm và gia tăng doanh thu.

Đây là một dự án Machine Learning mô phỏng bài toán thực tế trong Retail Analytics, tập trung vào việc phân tích hành vi khách hàng và dự đoán khả năng tái mua dựa trên dữ liệu tương tác sản phẩm. Dự án bao gồm toàn bộ quy trình từ Data Preprocessing, Feature Engineering, Label Construction, Model Training đến Dashboard Visualization.

Một điểm nổi bật của dự án là biến mục tiêu không được gán nhãn thủ công mà được xây dựng từ dữ liệu hành vi khách hàng thông qua phương pháp Information Content Weighting (ICW) kết hợp thuật toán Otsu Thresholding để xác định ngưỡng phân loại tối ưu.

Kết quả được trực quan hóa bằng Streamlit Dashboard, hỗ trợ theo dõi hành vi khách hàng, phân tích đặc trưng và hiển thị kết quả dự đoán một cách trực quan.

### Business Problem

Các doanh nghiệp bán lẻ thường gặp những câu hỏi quan trọng như:

* Khách hàng nào có khả năng quay lại mua hàng?
* Khách hàng nào đang có dấu hiệu rời bỏ nền tảng?
* Nhóm khách hàng nào nên được ưu tiên trong các chiến dịch marketing?
* Làm thế nào để cá nhân hóa trải nghiệm mua sắm dựa trên hành vi thực tế?

Dự án hướng đến việc hỗ trợ ra quyết định dựa trên dữ liệu (Data-Driven Decision Making), giúp doanh nghiệp xác định nhóm khách hàng tiềm năng và tối ưu hóa nguồn lực kinh doanh.

### Machine Learning Workflow

Dự án được triển khai theo quy trình thực tế:

1. Data Collection & Data Generation
2. Data Cleaning & Preprocessing
3. Exploratory Data Analysis (EDA)
4. Feature Engineering
5. Repurchase Label Construction
6. Machine Learning Model Training
7. Model Evaluation & Optimization
8. Dashboard Visualization & Decision Support

### Repurchase Label Construction

Một điểm nổi bật của dự án là biến mục tiêu không được gán nhãn thủ công mà được xây dựng trực tiếp từ dữ liệu hành vi khách hàng.

#### Step 1: Behavioral Condition Encoding

Các chỉ số hành vi được phân tích và chuyển đổi thành các điều kiện nhị phân dựa trên giá trị trung vị (Median - Q2).

#### Step 2: Information Content Weighting (ICW)

Mỗi điều kiện được gán trọng số dựa trên mức độ hiếm gặp của hành vi theo công thức:

IC Weight = -ln(p)

Trong đó:

* p là tỷ lệ khách hàng thỏa mãn điều kiện.
* Hành vi càng hiếm thì trọng số càng lớn.

#### Step 3: Full Score Calculation

Điểm hành vi tổng hợp được tính từ tổng các điều kiện có trọng số, phản ánh mức độ sẵn sàng tái mua của từng khách hàng.

#### Step 4: Otsu Thresholding

Thuật toán Otsu được sử dụng để tự động tìm ngưỡng phân tách tối ưu giữa hai nhóm khách hàng.

Kết quả cuối cùng:

* Repurchase = 1: Khách hàng có khả năng tái mua.
* Repurchase = 0: Khách hàng có khả năng không tái mua.

Toàn bộ quy trình được xây dựng theo hướng Data-Driven, hạn chế tối đa việc thiết lập ngưỡng dựa trên kinh nghiệm chủ quan.

### Technologies Used

* Python
* Pandas
* NumPy
* Scikit-learn
* XGBoost
* LightGBM
* Matplotlib
* Seaborn
* Streamlit
* Jupyter Notebook


---

## Tính năng chính

- Upload file CSV chứa dữ liệu khách hàng đã xử lý.
- Hiển thị tóm tắt dataset: số dòng, số khách hàng, số danh mục và số cột.
- Lọc theo `product_category` và `user_id` để chọn khách hàng cụ thể.
- Hiển thị các đặc trưng khách hàng dưới dạng bảng và hình tròn.
- Hiển thị kết quả tái mua dựa trên cột `repurchase_pred` hoặc `repurchase` trong file dữ liệu.

---

## Dữ liệu đầu vào

Dashboard `src/repurchase_prediction_dashboard.py` yêu cầu file CSV chứa tối thiểu các cột:

- `user_id`
- `product_category`
- `repurchase_pred` hoặc `repurchase`

Các cột bổ sung có thể có và được sử dụng để hiển thị dữ liệu chi tiết:

- `age`
- `total_view`, `total_click`, `total_cart`, `total_wishlist`
- `click_through_rate`, `cart_rate`
- `unique_brands`
- `avg_price`, `avg_discount`, `avg_purchase_value`, `avg_rating`
- `user_total_categories`, `category_share`
- `view_to_click_ratio`, `cart_to_click_ratio`, `wishlist_to_cart_ratio`
- `active_engagement_ratio`, `category_commitment_score`, `exploration_score`, `engagement_depth_score`, `discount_sensitivity`, `brand_loyalty_score`
- `click_engagement_rate`, `value_per_click`, `price_to_value_ratio`, `category_breadth_score`, `rating_concentration`

> Lưu ý: Dashboard vẫn chạy nếu không đủ tất cả cột trên, nhưng chỉ hiển thị dữ liệu của những cột tồn tại trong file.

---

## Chu trình dữ liệu trong repository

- `data/raw/retail_personalization_dataset.csv`: dữ liệu gốc cá nhân hóa bán lẻ.
- `data/processed/predictions_lgbm (1).csv`: Sau khi dùng thuật toán XGBooost để machine learning có chúa trường kết quả dự doán cho máy học

---

## Cài đặt

1. Tạo môi trường ảo và kích hoạt:

```bash
python -m venv venv
venv\Scripts\activate
```

2. Cài đặt thư viện:

```bash
pip install -r requirements.txt
```

---

## Chạy dashboard

```bash
streamlit run src/repurchase_prediction_dashboard.py
```

Sau khi dashboard chạy, upload file CSV vào sidebar và chọn `Category` + `User ID` để tải thông tin khách hàng.

---

## Cấu trúc thư mục

```
Retail-Customer-Churn-Analysis-master/
├── README.md
├── requirements.txt
├── data/
│   ├── raw/
│      └── retail_personalization_dataset.csv
│   └── processed/
│       ├── confusion_matrix_test.png
│       ├── derived_features (1).csv
│       ├── feature_data (1).csv
│       ├── predictions_lgbm (1).csv
│       ├── predictions_lr (1).csv
│       ├── predictions_xgb (1).csv
│       ├── processed_data (1).csv
│       └── synthetic_data (1).csv
├── models/
│   ├── best_churn_model.joblib
│   ├── best_model.pkl
│   ├── category_encoder.joblib
│   ├── category_label_encoder.joblib
│   ├── feature_columns.joblib
│   ├── feature_importance.csv
│   ├── lr_model.pkl
│   ├── lr_optimal_threshold.pkl
│   ├── next_product_category_lgbm_model.joblib
│   ├── next_product_category_rf_model.joblib
│   ├── optimal_threshold.pkl
│   ├── test_data.pkl
│   ├── test_product_category_model.py
│   ├── train_model.py
│   ├── xgb_model.pkl
│   └── xgb_optimal_threshold.pkl
├── notebooks/
│   ├── 1_Data_Generation.ipynb
│   ├── 2_Exploratory_Data_Analysis.ipynb
│   ├── 3_Model_Building.ipynb
│   └── 4_Model_Evaluation.ipynb
└── src/
    ├── app.py
    └── repurchase_prediction_dashboard.py
```

---

## Ghi chú

- `src/repurchase_prediction_dashboard.py` là dashboard chính của dự án.
- `src/app.py` tồn tại nhưng không phải luồng chính trong README này.
- `models/next_product_category_rf_model.joblib` và `models/category_encoder.joblib` được load trong `repurchase_prediction_dashboard.py`, nhưng hiện tại dashboard chỉ dùng nhãn `repurchase_pred`/`repurchase` từ dữ liệu.
- Nếu muốn chạy các notebook, hãy mở file trong `notebooks/` và thực hiện theo thứ tự.

---


## Dashboard Predict Category Customer



<img width="1919" height="1023" alt="image" src="https://github.com/user-attachments/assets/88204c43-7093-4f50-a423-3c9156d47bea" />

<img width="1919" height="1029" alt="image" src="https://github.com/user-attachments/assets/381c0c09-53c8-426c-8c2a-5ad3afb306cd" />

<img width="1913" height="1021" alt="image" src="https://github.com/user-attachments/assets/0ee5cfb3-19c7-4484-a4ed-f2e42b51c3a8" />

<img width="1917" height="1023" alt="image" src="https://github.com/user-attachments/assets/9787c8ff-fed3-4ed4-b072-9858e6d1ef8b" />

<img width="1914" height="1031" alt="image" src="https://github.com/user-attachments/assets/59797079-33dc-4a6b-940d-af5ae1eca535" />

<img width="1919" height="1021" alt="image" src="https://github.com/user-attachments/assets/0b1f91ec-7b50-44a6-8af4-81da0c7b168c" />








