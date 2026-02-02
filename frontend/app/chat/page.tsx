"use client";
import { useState, useEffect, useRef } from "react";
import { supabase } from "@/lib/supabase";
import { useRouter } from "next/navigation";
import { Send, LogOut } from "lucide-react";

export default function Chat() {
  const [messages, setMessages] = useState<any[]>([]);
  const [input, setInput] = useState("");
  const [loading, setLoading] = useState(false);
  const [user, setUser] = useState<any>(null);
  const [chatId, setChatId] = useState<string>("");
  const scrollRef = useRef<HTMLDivElement>(null);
  const router = useRouter();

  useEffect(() => {
    const init = async () => {
      const { data: { user } } = await supabase.auth.getUser();
      if (!user) return router.push("/");
      setUser(user);

      let { data: chats } = await supabase.from('chats').select('id').eq('user_id', user.id).limit(1);
      let cid = chats?.[0]?.id;
      if (!cid) {
         const { data: newChat } = await supabase.from('chats').insert({ user_id: user.id }).select().single();
         cid = newChat.id;
      }
      setChatId(cid);

      const { data: msgs } = await supabase.from('messages').select('*').eq('chat_id', cid).order('created_at', { ascending: true });
      if (msgs) setMessages(msgs);
    };
    init();
  }, []);

  useEffect(() => scrollRef.current?.scrollIntoView({ behavior: 'smooth' }), [messages]);

  const send = async () => {
    if (!input.trim()) return;
    const txt = input; 
    setInput("");
    setMessages(prev => [...prev, { role: "user", content: txt }]);
    setLoading(true);

    try {
      const res = await fetch(`${process.env.NEXT_PUBLIC_API_URL}/chat`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ user_id: user.id, chat_id: chatId, message: txt })
      });
      const data = await res.json();
      setMessages(prev => [...prev, { role: "assistant", content: data.response }]);
    } catch (e) { console.error(e); }
    setLoading(false);
  };

  return (
    <div className="flex flex-col h-screen bg-gray-50 md:max-w-md md:mx-auto shadow">
      <div className="p-4 bg-white shadow flex justify-between">
        <h2 className="font-bold">MindSpace</h2>
        <LogOut size={20} className="text-gray-500 cursor-pointer" onClick={() => {supabase.auth.signOut(); router.push("/");}}/>
      </div>
      <div className="flex-1 overflow-y-auto p-4 space-y-4">
        {messages.map((m, i) => (
          <div key={i} className={`flex ${m.role === 'user' ? 'justify-end' : 'justify-start'}`}>
            <div className={`p-3 rounded-2xl max-w-[85%] ${m.role === 'user' ? 'bg-blue-600 text-white' : 'bg-white border'}`}>{m.content}</div>
          </div>
        ))}
        {loading && <div className="text-gray-400 text-sm">Typing...</div>}
        <div ref={scrollRef} />
      </div>
      <div className="p-4 bg-white border-t flex gap-2">
        <input className="flex-1 bg-gray-100 p-3 rounded-full outline-none" value={input} onChange={e => setInput(e.target.value)} onKeyDown={e => e.key === 'Enter' && send()}/>
        <button onClick={send} className="bg-blue-600 text-white p-3 rounded-full"><Send size={20}/></button>
      </div>
    </div>
  );
}