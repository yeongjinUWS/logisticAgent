import React, { useState } from "react";
import "./TraningCSS.css";
import TraningAPI from "./TraningAPI";

export default function Traning() {

    const {
        file, setFile,
        columns,
        loading,
        samples,
        rowCount,
        handleUpload,
        selectedColumns,
        handleCheckboxChange,
        handleLearning,
        modelList,
        viewDetail, setViewDetail
    } = TraningAPI();

    return (
        <div className="container">
            {/* 좌측 */}
            <div className="left">
                <h2>📂 데이터 업로드</h2>

                <input
                    type="file"
                    accept=".xlsx,.xls,.csv"
                    onChange={(e) => setFile(e.target.files[0])}
                />

                <button onClick={handleUpload} disabled={!file || loading}>
                    {loading ? "Agent 분석 중..." : "데이터 분석"}
                </button>

                {file && (
                    <div className="file-info">
                        <strong>파일명:</strong> {file.name}
                    </div>
                )}

                <h2>학습된 Agent 모델</h2>

                {modelList.length === 0 ? (
                    <p className="empty">아직 학습된 모델이 없습니다.</p>
                ) : (
                    <ul className="model-list">
                        {modelList.map((model, index) => (
                            <div key={index} className="model-card">
                                <div className="model-header"
                                    onClick={() => {
                                        if (viewDetail === model) {
                                            setViewDetail()
                                        } else {
                                            setViewDetail(model)
                                        }
                                    }
                                    }
                                >
                                    <span className="model-name">{model.modelFile}</span>
                                </div>
                                {
                                    viewDetail === model ?
                                        <div className="model-body">
                                            <p><strong>대상:</strong> {model.title}</p>
                                            <p><strong>컬럼:</strong></p>
                                            <ul className="column-list">
                                                {model.encodedColumns.map((col, idx) => (
                                                    <li key={idx}>{col}</li>
                                                ))}
                                            </ul>
                                            <p className="date">
                                                생성일: {model.createdAt}
                                            </p>
                                        </div>

                                        :
                                        null
                                }
                            </div>
                        ))}
                    </ul>
                )}
            </div>

            {/* 우측 */}
            <div className="right">
                <h2>🤖 Agent 분석 결과</h2>

                {columns.length === 0 ? (
                    <div className="empty">
                        아직 분석된 데이터가 없습니다.
                    </div>
                ) : (
                    <div>
                        <text>
                            Data Count : {rowCount}
                        </text>
                        <div
                            className="flexTab"
                        >
                            <table>
                                <thead>
                                    <tr>
                                        <th>컬럼명</th>
                                        <th>학습</th>
                                    </tr>
                                </thead>
                                <tbody>
                                    {columns.map((col, idx) => (
                                        <tr key={idx}
                                            onClick={() => handleCheckboxChange(col)}
                                            onMouseOver={(e) => e.currentTarget.style.backgroundColor = '#f1faff'}
                                            onMouseOut={(e) => e.currentTarget.style.backgroundColor = 'transparent'}
                                        >
                                            <td>{col}</td>
                                            <td>
                                                <input
                                                    type="checkbox"
                                                    checked={selectedColumns.has(col)}
                                                    onChange={() => handleCheckboxChange(col)}
                                                />
                                            </td>
                                        </tr>
                                    ))}
                                </tbody>
                            </table>
                            <table >
                                <thead >
                                    <tr>
                                        {columns.map((colName, index) => (
                                            <th key={index}
                                                className="headText" >{colName} </th>
                                        ))}
                                    </tr>
                                </thead>
                                <tbody>
                                    {samples.map((row, rowIndex) => (
                                        <tr key={rowIndex} >
                                            {columns.map((colName, colIndex) => (
                                                <td key={colIndex}
                                                    className="bodyText"
                                                >
                                                    {row[colName] || "-"}
                                                </td>
                                            ))}
                                        </tr>
                                    ))}
                                </tbody>
                            </table>
                        </div>

                        <button className="button"
                            onClick={() => handleLearning()}
                        >
                            업로드 및 모델 학습
                        </button>

                    </div>
                )}
            </div>
        </div>
    );
}