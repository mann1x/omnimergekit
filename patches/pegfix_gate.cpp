#include "chat.h"
#include "chat-peg-parser.h"
#include <cstdio>
#include <cstdlib>
#include <string>
#include <functional>
int main() {
    common_chat_parser_params pp;
    pp.format = COMMON_CHAT_FORMAT_PEG_NATIVE;
    // grammar demands a literal the input never contains -> forces result.fail()
    std::function<common_peg_parser(common_chat_peg_builder&)> mk = [](common_chat_peg_builder & p) {
        return p.literal("ZZ_NEVER_PRESENT_ZZ") + p.end();
    };
    common_peg_arena arena = build_chat_peg_parser(mk);
    std::string bad1 = "<think>\nThe user is asking whether hamburgers are sandwiches";
    std::string bad2 = std::string("# ") + "\xef\xbf\xbd" + " The Grandiose Session Plan";
    int fails = 0;
    for (int strict = 0; strict <= 1; ++strict) {
        if (strict) setenv("LLAMA_CHAT_PEG_STRICT","1",1); else unsetenv("LLAMA_CHAT_PEG_STRICT");
        for (int w = 0; w < 2; ++w) {
            const std::string & in = w ? bad2 : bad1;
            bool threw=false; std::string content;
            try { auto m = common_chat_peg_parse(arena, in, false, pp); content = m.content; }
            catch (const std::exception & e) { threw = true; }
            printf("  strict=%d trigger=%d -> %-8s content_bytes=%zu\n", strict, w+1, threw?"THREW":"returned", content.size());
            if (strict && !threw) { puts("    FAIL strict must throw"); fails++; }
            if (!strict && threw) { puts("    FAIL lenient must not throw"); fails++; }
            if (!strict && !threw && content.empty()) { puts("    FAIL lenient returned EMPTY"); fails++; }
        }
    }
    printf(fails?"PEGFIX_GATE_FAIL %d\n":"PEGFIX_GATE_PASS\n", fails);
    return fails?1:0;
}
