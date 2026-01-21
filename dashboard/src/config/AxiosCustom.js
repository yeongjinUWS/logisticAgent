import axios from "axios";

const baseURL = "http://localhost:8080";

const AxiosCustom = axios.create({
    baseURL: baseURL,
    headers: { "Content-Type": "application/json" },
});


export default AxiosCustom;