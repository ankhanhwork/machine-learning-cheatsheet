# ML Study Library

Website tĩnh để đọc tài liệu Markdown và Jupyter Notebook trong repository, đồng thời tải bản gốc về máy. Site không cần database hay API key.

## Chạy thử trên máy

Cần Node.js 20 trở lên.

```bash
npm run dev
```

Mở địa chỉ `http://localhost:4173`. Lệnh build tạo website đã đóng gói trong `dist/`:

```bash
npm run build
```

## Deploy lên Vercel

1. Push repository này lên GitHub.
2. Trong Vercel chọn **Add New → Project** rồi import repository.
3. Để Framework Preset là **Other**. Build Command là `npm run build`, Output Directory là `dist` (đã khai báo trong `vercel.json`). Không cần biến môi trường.
4. Chọn **Deploy**.

Mỗi lần push commit mới, Vercel sẽ build và cập nhật website. Không cần đưa thư mục `dist/` lên GitHub.

## Thêm tài liệu

Build tự tìm và đóng gói file `.md`, `.ipynb`, `.pdf`, `.docx`, `.csv` và `.py` trong repository, trừ các file nằm trong `.git`, `node_modules`, `dist`, `generator`, `.docx-render` và `.vercel`. Markdown và notebook đọc trực tiếp trên web; notebook giữ nguyên file JSON gốc để tải xuống. PDF có thể xem trực tiếp; các loại khác có thể tải xuống.

Tài liệu trong repository sẽ được đóng gói cùng site và ai truy cập được website cũng có thể tải chúng. Chỉ commit tài liệu bạn muốn đưa lên website.
