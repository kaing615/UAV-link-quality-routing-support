Write-Host "Bắt đầu cài đặt AWS CLI và Terraform bằng Winget..." -ForegroundColor Cyan

Write-Host "Đang cài đặt AWS CLI..." -ForegroundColor Yellow
winget install --id Amazon.AWSCLI -e --accept-package-agreements --accept-source-agreements

Write-Host "Đang cài đặt Terraform..." -ForegroundColor Yellow
winget install --id Hashicorp.Terraform -e --accept-package-agreements --accept-source-agreements

Write-Host ""
Write-Host "Cài đặt hoàn tất!" -ForegroundColor Green
Write-Host "LƯU Ý QUAN TRỌNG: Bạn CẦN ĐÓNG và MỞ LẠI cửa sổ VS Code / Terminal này để hệ thống nhận diện lệnh 'aws' và 'terraform'." -ForegroundColor Red
Write-Host "Sau khi mở lại, hãy gõ lệnh sau để đăng nhập vào AWS:" -ForegroundColor Cyan
Write-Host "aws configure" -ForegroundColor Yellow
