# Customer Repurchase Prediction Dashboard

## Tổng quan

Đây là một dự án dashboard Streamlit cho phân tích hành vi khách hàng và hiển thị kết quả dự đoán tái mua theo cặp `(user_id, product_category)`.

Dashboard chính trong dự án là: `src/repurchase_prediction_dashboard.py`.

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
- `data/raw/online_retail_II.csv`: dữ liệu nguồn ban đầu khác của dự án.
- `data/processed/predictions_lgbm (1).csv`: tệp dữ liệu đã tạo sẵn chứa nhãn dự đoán tái mua, phù hợp để upload vào dashboard.

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
│   │   ├── online_retail_II.csv
│   │   └── retail_personalization_dataset.csv
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
│   ├── preprocessor.pkl
│   ├── product_category_model.joblib
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

## Phần mở rộng

- Nếu cần, có thể bổ sung logic dự đoán trực tiếp bằng model đã load.
- Nếu bạn dùng file `predictions_lgbm.csv` khác tên, hãy đổi tên hoặc sửa đường dẫn trong README/ notebook cho phù hợp.

---

## Ảnh demo

Phần này dành để chèn ảnh minh họa dashboard. Bạn có thể thêm ảnh vào repository và dùng đường dẫn bên dưới.

![Ảnh demo dashboard](src/dashboard_preview.png)

*Ghi chú: ảnh sẽ hiển thị khi file ảnh `src/dashboard_preview.png` được thêm vào repo.*

---
