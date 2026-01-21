import { useState } from "react";
import AxiosCustom from "../../config/AxiosCustom";

export default function ViewAPI() {
    const [messages, setMessages] = useState([]);

    const onSend = async (text) => {
        if (!text.trim()) return;
        const myMsg = { id: Date.now(), text: text, sender: 'user' };
        setMessages(prev => [...prev, myMsg]);
        
        console.log(myMsg)
        try {
            AxiosCustom.post('/api/chat', {
                message: text
            }).then((response) => {
                console.log(response)
                setMessages(prev => [...prev, { id: response.data.id || Date.now(), text: response.data.result, sender: 'bot' }]);
            }).catch((error) => {
                console.log(error);
            })

        } catch (error) {
            console.error("Chat Error:", error);
        }
    };

    return {
        messages,
        onSend,
        setMessages
    };
}