import os
import pandas as pd
import tkinter as tk
from tkinter import filedialog, ttk, messagebox
from config import RIGHT_HAND_KEYS

class DatasetFileAnalyzer:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("EMG 데이터셋 파일 분석기")
        self.root.geometry("500x650")

        # 파일 선택 섹션
        self.btn_select = tk.Button(self.root, text="분석할 CSV 파일 선택 (다중 선택 가능)", command=self.select_files)
        self.btn_select.pack(pady=10)

        self.lbl_info = tk.Label(self.root, text="선택된 파일: 없음", wraplength=450, fg="gray")
        self.lbl_info.pack(pady=5)

        # 통계 요약 레이블
        self.lbl_summary = tk.Label(self.root, text="", justify="left", font=("Arial", 10))
        self.lbl_summary.pack(pady=10)

        # 결과 테이블 (Treeview)
        self.tree = ttk.Treeview(self.root, columns=("Key", "Count"), show="headings")
        self.tree.heading("Key", text="키 명칭")
        self.tree.heading("Count", text="입력 횟수")
        self.tree.column("Key", anchor="center", width=200)
        self.tree.column("Count", anchor="center", width=200)
        self.tree.pack(expand=True, fill="both", padx=10, pady=10)

    def select_files(self):
        """개별 파일 또는 다중 파일 선택 및 분석 실행"""
        file_paths = filedialog.askopenfilenames(
            title="CSV 파일 선택",
            filetypes=[("CSV Files", "*.csv"), ("All Files", "*.*")]
        )
        
        if not file_paths:
            return

        file_count = len(file_paths)
        self.lbl_info.config(text=f"선택된 파일: {file_count}개")
        self.analyze_files(file_paths)

    def analyze_files(self, file_paths):
        """선택된 파일 리스트를 순회하며 통계 계산"""
        inv_map = {v: k for k, v in RIGHT_HAND_KEYS.items()}
        
        total_stats = {}
        total_samples = 0
        total_presses = 0

        # 기존 테이블 데이터 초기화
        for item in self.tree.get_children():
            self.tree.delete(item)

        for file in file_paths:
            try:
                df = pd.read_csv(file)
                total_samples += len(df)
                
                # Event 열에서 0이 아닌 값(입력 시점) 필터링
                events = df[df['Event'] != 0]['Event'].tolist()
                
                for ev in events:
                    key_name = inv_map.get(ev, f"Unknown({ev})")
                    total_stats[key_name] = total_stats.get(key_name, 0) + 1
                    total_presses += 1
            except Exception as e:
                print(f"파일 처리 오류 ({os.path.basename(file)}): {e}")

        # 요약 정보 업데이트
        summary_text = (
            f"분석 대상 파일: {len(file_paths)}개\n"
            f"전체 샘플 수: {total_samples:,}개 (약 {total_samples/400:.1f}초)\n"
            f"전체 키 입력: {total_presses:,}회"
        )
        self.lbl_summary.config(text=summary_text)

        # 정렬 및 테이블 삽입
        sorted_stats = sorted(total_stats.items(), key=lambda x: x[1], reverse=True)
        for key, count in sorted_stats:
            self.tree.insert("", "end", values=(key, count))

    def run(self):
        self.root.mainloop()

if __name__ == "__main__":
    app = DatasetFileAnalyzer()
    app.run()