import { useEffect, useState } from "react";
import AxiosCustom from "../../config/AxiosCustom";

export default function TraningAPI() {
    const [file, setFile] = useState(null);
    const [columns, setColumns] = useState([]);
    const [samples, setSamples] = useState([]);
    const [loading, setLoading] = useState(false);
    const [rowCount, setRowCount] = useState(0);
    const [messages, setMessages] = useState([]);
    const [selectedColumns, setSelectedColumns] = useState(new Set());
    const [modelList, setModelList] = useState([]);
    const [viewDetail, setViewDetail] = useState();
    const [analyze,setAnalyze] = useState(null);
    useEffect(() => {
        AxiosCustom.post('/api/getModels')
            .then((response) => {
                console.log(response.data.result);
                setModelList(response.data.result);
            })
            .catch((error) => {
                console.log(error);
            })
    }, [])

    const handleUpload = async () => {
        if (!file) return;
        setLoading(true);
        try {
            const formData = new FormData();
            formData.append("file", file);
            AxiosCustom.post('/api/upload',
                formData
                ,
                {
                    headers: {
                        "Content-Type": "multipart/form-data",
                    },
                }).then((response) => {
                    console.log(response)
                    setColumns(response.data.columns);
                    setSamples(response.data.samples);
                    setRowCount(response.data.rowCount);
                    setAnalyze(response.data.analyze);
                }).catch((error) => {
                    console.log(error);
                })

        } catch (error) {
            console.error("Chat Error:", error);
        }
        setLoading(false);
    };
    const handleCheckboxChange = (colName) => {
        const newSelection = new Set(selectedColumns);
        if (newSelection.has(colName)) {
            newSelection.delete(colName); // 이미 있으면 제거
        } else {
            newSelection.add(colName);    // 없으면 추가
        }
        setSelectedColumns(newSelection);
    };

    function handleLearning() {
        if (!file) return;
        try {
            const formData = new FormData();
            formData.append("file", file);
            formData.append("column", (Array.from(selectedColumns)));
            formData.append('category',analyze.category);
            formData.append('target_recommendation',analyze.target_recommendation);
            formData.append('description',analyze.description);
            console.log( JSON.stringify(samples))
            formData.append('samples', JSON.stringify(samples));
            AxiosCustom.post('/api/learning',
                formData
                ,
                {
                    headers: {
                        "Content-Type": "multipart/form-data",
                    },
                }).then((response) => {
                    console.log(response)

                }).catch((error) => {
                    console.log(error);
                })

        } catch (error) {
            console.error("Chat Error:", error);
        }
    };
    return {
        file, setFile,
        columns,
        loading,
        messages,
        samples,
        rowCount,
        selectedColumns,
        handleCheckboxChange,
        handleUpload,
        handleLearning,
        modelList,
        viewDetail, setViewDetail,
        analyze
    };
}