import asyncio
import logging
from kag.common.conf import KAG_CONFIG
from kag.common.registry import import_modules_from_path
from kag.interface import SolverPipelineABC
from kag.solver.reporter.trace_log_reporter import TraceLogReporter

logger = logging.getLogger(__name__)


class VkbDemo:
    """
    init for kag client
    """

    async def qa(self, query):
        reporter: TraceLogReporter = TraceLogReporter()
        resp = SolverPipelineABC.from_config(KAG_CONFIG.all_config["kag_solver_pipeline"])
        answer = await resp.ainvoke(query, reporter=reporter)

        logger.info(f"\n\nso the answer for '{query}' is: {answer}\n\n")

        info, status = reporter.generate_report_data()
        logger.info(f"trace log info: {info.to_dict()}")
        return answer


if __name__ == "__main__":
    import_modules_from_path("./prompt")

    demo = VkbDemo()
    query = """
    public function loginByUsername($username, $password) {
        $stmt = $this->connection->prepare("SELECT user.iduser, user.firstname, user.lastname, user.username, user.password, user.type, profile.img 
                                           FROM user 
                                           JOIN profile ON user.iduser = profile.user_iduser 
                                           WHERE user.username = ? AND user.password = ?");
        $stmt->bind_param("ss", $username, $password);
        $stmt->execute();
        $results = $stmt->get_result();

        $user_list = [];
        while ($row = $results->fetch_assoc()) {
            $user = [
                'iduser' => $row['iduser'],
                'username' => $row['username'],
                'img' => $row['img'],
                'password' => $row['password'],
                'firstname' => $row['firstname'],
                'lastname' => $row['lastname'],
                'type' => $row['type']
            ];
            $user_list[] = $user;
        }
        return $user_list;
    }
    请告诉我这段代码是否存在漏洞"""






    answer = asyncio.run(demo.qa(query))
    print(f"Question: {query}")
    print(f"Answer: {answer}")
